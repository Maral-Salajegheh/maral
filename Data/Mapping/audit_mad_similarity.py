#!/usr/bin/env python3
"""Read-only, local audit of MAD copies inside sibling MasterIndex PDFs.

Dependencies: numpy, Pillow, pyarrow (no GPU, OCR, ImageHash or OpenCV).
Run from the life-docai repository root:
  pixi run -e dev python audit_mad_similarity.py --stack-id mklia000000024690
Targets have successful rendered pages in the supplied inventory and MORE
physical pages than mapping rows. CSV alignment status is reported, not used
as a selection gate. Only same-stack rendered MAD references are compared.
Place this script beside 00_map_mid_to_adb.py in datasets/mapping.

Outputs are review suggestions, NOT approved matches or corrected labels.
No input, image, mapping, or training file is modified. No network requests.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps


def arguments():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    here = Path(__file__).resolve().parent
    default_root = next((p for p in (here, *here.parents, Path.cwd(), *Path.cwd().parents)
                         if (p / 'packages/life-docai-analytics').is_dir()), Path.cwd())
    p.add_argument('--repo-root', type=Path, default=default_root)
    p.add_argument('--batch-id', default='antrag_life_A00_G07_MAD')
    p.add_argument('--mapping', type=Path, help='life_mid_pages.csv (not predictions or stack_pages).')
    p.add_argument('--inventory', type=Path, help='page_inventory.parquet from the render batch.')
    p.add_argument('--stack-id', action='append', help='Repeat to restrict stacks; targets must have extra rendered pages.')
    p.add_argument('--top-k', type=int, default=3, help='Best candidates per reference page AND target MID, including poor matches.')
    p.add_argument('--phash-distance', type=int, default=12, help='Exploratory review cutoff out of 63 bits; NOT a validated threshold.')
    p.add_argument('--rotations', default='0,180', help='Reference rotations to try, e.g. 0,90,180,270.')
    p.add_argument('--path-prefix', action='append', default=[], metavar='OLD=NEW', help='Translate inventory image paths when moved to another server.')
    p.add_argument('--output-dir', type=Path, help='NEW directory only; existing directories are never overwritten.')
    a = p.parse_args()
    if a.top_k < 1 or not 0 <= a.phash_distance <= 63:
        p.error('--top-k must be positive and --phash-distance must be 0..63.')
    try:
        a.rotations = list(dict.fromkeys(int(x) for x in a.rotations.split(',')))
        if not a.rotations or any(x not in (0, 90, 180, 270) for x in a.rotations):
            raise ValueError
        a.prefixes = [x.split('=', 1) for x in a.path_prefix]
        if any(len(x) != 2 or not x[0] for x in a.prefixes):
            raise ValueError
    except ValueError:
        p.error('Invalid rotations or OLD=NEW prefix.')
    base = a.repo_root / 'packages/life-docai-analytics/life_docai_analytics_antrag/datasets'
    a.mapping = a.mapping or (here / 'output/life_mid_pages.csv' if (here / 'output/life_mid_pages.csv').is_file()
                              else base / 'mapping/output/life_mid_pages.csv')
    a.inventory = a.inventory or base / 'aws_download_ingestion/output' / a.batch_id / 'page_inventory.parquet'
    a.output_dir = a.output_dir or a.mapping.parent / ('mad_similarity_audit_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ'))
    return a


def normalized(row):
    return {str(k).strip().lower(): str(v or '').strip() for k, v in row.items() if k is not None}


def read_mapping(path, selected):
    groups = defaultdict(lambda: {'rows': [], 'classes': set(), 'statuses': set()})
    with path.open(newline='', encoding='utf-8-sig') as f:
        sample = f.read(65536)
        f.seek(0)
        try:
            dialect = csv.Sniffer().sniff(sample, delimiters=',;\t|')
        except csv.Error:
            dialect = csv.excel
        reader = csv.DictReader(f, dialect=dialect)
        names = {x.strip().lower() for x in (reader.fieldnames or [])}
        if not {'stack_id', 'masterindex_id', 'pdf_page_number'} <= names or not ({'sst', 'page_class'} & names):
            raise ValueError('Mapping must contain stack_id, masterindex_id, pdf_page_number and sst or page_class.')
        for raw in reader:
            r = normalized(raw)
            stack, mid = r['stack_id'].lower(), r['masterindex_id']
            if not stack or not mid:
                continue
            g = groups[(stack, mid)]
            g['rows'].append(r)
            label = (r.get('sst') or r.get('page_class', '')).upper()
            if label:
                g['classes'].add(label)
            g['statuses'].add(r.get('mid_alignment_status', ''))
    if selected:
        # Do not silently merge stack identities or strip meaningful zeroes.
        groups = {key: value for key, value in groups.items() if key[0] in selected}
    return groups


def reference_pages(group, pages):
    """Use MAD-specific pages, not an entire mixed MID as a reference."""
    if group['classes'] == {'MAD'}:
        return pages
    positions = {r['pdf_page_number'] for r in group['rows']
                 if (r.get('page_class') or r.get('sst', '')).upper() == 'MAD'
                 and r.get('mid_alignment_status') in ('ALIGNED', 'RESOLVED_FOREIGN')}
    return [r for r in pages if str(r['source_page_number']) in positions]


def read_inventory(path, mids):
    import pyarrow.parquet as pq
    parquet = pq.ParquetFile(path)
    required = {'masterindex_id', 'source_page_number', 'image_path', 'status'}
    available = set(parquet.schema_arrow.names)
    if not required <= available:
        raise ValueError('Missing inventory columns: ' + ', '.join(sorted(required - available)))
    columns = sorted(required | ({'image_sha256', 'quality_status', 'corpus_document_page_count'} & available))
    result = defaultdict(list)
    seen = set()
    for batch in parquet.iter_batches(columns=columns, batch_size=32768):
        for r in batch.to_pylist():
            mid = str(r['masterindex_id'] or '').strip()
            if mid not in mids or r['status'] != 'success' or r['source_page_number'] is None:
                continue
            r['masterindex_id'] = mid
            r['source_page_number'] = int(r['source_page_number'])
            key = (mid, r['source_page_number'])
            if key in seen:
                raise ValueError(f'Duplicate inventory MID/page: {key}')
            seen.add(key)
            result[mid].append(r)
    for pages in result.values():
        pages.sort(key=lambda r: r['source_page_number'])
    return result


# Orthonormal DCT-II: pHash uses 8x8 low frequencies, excluding DC (63 bits).
_x = np.arange(32)
_dct = np.cos(np.pi * np.arange(32)[:, None] * (2 * _x + 1) / 64) * np.sqrt(2 / 32)
_dct[0] /= np.sqrt(2)


def features(image):
    gray = image.convert('L')
    small = np.asarray(gray.resize((32, 32), Image.Resampling.LANCZOS), dtype=np.float64)
    low = (_dct @ small @ _dct.T)[:8, :8].ravel()[1:]
    bits = low > np.median(low)
    h = sum(int(bit) << i for i, bit in enumerate(bits))
    detail = np.asarray(gray.resize((256, 384), Image.Resampling.LANCZOS), dtype=np.float32) / 255
    ink = float((detail < 0.85).mean())
    low_info = float(detail.std()) < 0.02 or ink < 0.002 or ink > 0.98
    return h, detail, low_info


def resolve_path(raw, args):
    value = str(raw or '')
    for old, new in args.prefixes:
        if value == old or value.startswith(old.rstrip('/') + '/'):
            value = new.rstrip('/') + value[len(old.rstrip('/')):]
            break
    path = Path(value)
    return path if path.is_absolute() else args.repo_root / path


def load_page(row, args, rotations, errors, stack):
    path = resolve_path(row.get('image_path'), args)
    try:
        with Image.open(path) as src:
            image = ImageOps.exif_transpose(src).convert('L')
            values = {angle: features(image.rotate(angle, expand=True)) for angle in rotations}
        return dict(row, resolved_image_path=str(path), features=values)
    except Exception as error:
        errors.writerow({'stack_id': stack, 'masterindex_id': row['masterindex_id'],
                         'pdf_page_number': row['source_page_number'], 'image_path': str(path),
                         'error': f'{type(error).__name__}: {error}'})
        return None


def detail_scores(reference, target):
    """Small translation search on normalized grayscale images; no semantic claim.

    Compare common overlaps only. Return best ink F1, then grayscale MAE at
    that shift. Neither score is a calibrated match probability.
    """
    best = (-1.0, -1.0, 0, 0)
    h, w = reference.shape
    for dy in range(-8, 9, 2):
        for dx in range(-8, 9, 2):
            a = reference[max(0, dy):min(h, h + dy), max(0, dx):min(w, w + dx)]
            b = target[max(0, -dy):min(h, h - dy), max(0, -dx):min(w, w - dx)]
            ai, bi = a < 0.85, b < 0.85
            denom = int(ai.sum()) + int(bi.sum())
            f1 = 2 * int((ai & bi).sum()) / denom if denom else 0.0
            mae = float(np.abs(a - b).mean())
            score = (f1, -mae, dx, dy)
            if score[:2] > best[:2]:
                best = score
    return best[0], -best[1], best[2], best[3]


def mapped_rows(group, page):
    # These are CURRENT mapping claims, not independently verified truth.
    return json.dumps([{k: r.get(k, '') for k in
                       ('image_id', 'process_id', 'doc_id', 'subdoc_idx', 'sst', 'page_class',
                        'sfdoc_class', 'seqno', 'pdf_page_source', 'mid_alignment_status')}
                      for r in group['rows'] if r['pdf_page_number'] == str(page)], ensure_ascii=False)


def writer(folder, name, fields):
    f = (folder / name).open('x', newline='', encoding='utf-8-sig')
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader()
    return f, w


def main():
    args = arguments()
    for path in (args.mapping, args.inventory):
        if not path.is_file():
            raise FileNotFoundError(f'Input not found: {path}. Use --repo-root, --mapping or --inventory.')
    selected = {s.strip().lower() for s in (args.stack_id or [])}
    groups = read_mapping(args.mapping, selected)
    mid_stacks = defaultdict(set)
    stacks = defaultdict(dict)
    for (stack, mid), group in groups.items():
        mid_stacks[mid].add(stack)
        stacks[stack][mid] = group
    ambiguous_mids = {mid for mid, values in mid_stacks.items() if len(values) > 1}
    # Inventory is read BEFORE choosing targets: mapping may predate rendering.
    inventory = read_inventory(args.inventory, set(mid_stacks))
    jobs = []
    skipped = []
    scope_rows = []
    for stack in sorted(selected - set(stacks)):
        skipped.append({'stack_id': stack, 'status': 'STACK_NOT_FOUND_IN_MAPPING'})
    for stack, mids in sorted(stacks.items()):
        if any(mid in ambiguous_mids for mid in mids):
            skipped.append({'stack_id': stack, 'status': 'AMBIGUOUS_MID_STACK_ASSIGNMENT'})
            continue
        targets = []
        for mid, group in mids.items():
            pages = inventory.get(mid, [])
            expected = {int(r['corpus_document_page_count']) for r in pages
                        if r.get('corpus_document_page_count') is not None}
            complete = not expected or (len(expected) == 1 and
                        {r['source_page_number'] for r in pages} == set(range(1, next(iter(expected)) + 1)))
            extra = len(pages) - len(group['rows'])
            if not pages:
                decision = 'NOT_IN_RENDERED_SAMPLE'
            elif not complete:
                decision = 'INCOMPLETE_RENDER_INVENTORY'
            elif group['classes'] == {'MAD'}:
                decision = 'MAD_REFERENCE_MID'
            elif extra > 0:
                decision = 'EXTRA_PAGES_TARGET'
                targets.append(mid)
            elif extra == 0:
                decision = 'NO_EXTRA_RENDERED_PAGES'
            else:
                decision = 'FEWER_RENDERED_PAGES_THAN_MAPPING_ROWS'
            scope_rows.append(dict(stack_id=stack, masterindex_id=mid,
                mapping_rows=len(group['rows']), successful_rendered_pages=len(pages),
                extra_rendered_pages=extra if pages else '',
                mapping_alignment_status='|'.join(sorted(group['statuses'])), decision=decision))
        if not targets:
            skipped.append({'stack_id': stack, 'status': 'NO_RENDERED_EXTRA_PAGE_TARGET'})
            continue
        refs = [mid for mid, g in mids.items()
                if any(target != mid for target in targets)
                and reference_pages(g, inventory.get(mid, []))]
        if refs and targets:
            jobs.append((stack, refs, targets))
        else:
            skipped.append({'stack_id': stack, 'status': 'NO_MAD_REFERENCE' if not refs else 'NO_OTHER_TARGET_MID'})
    args.output_dir.mkdir(parents=True, exist_ok=False)
    common = ['stack_id', 'main_masterindex_id', 'main_pdf_page_number', 'mad_masterindex_id',
              'mad_pdf_page_number', 'candidate_rank', 'phash_distance_63bits', 'within_review_cutoff',
              'reference_rotation_degrees', 'ink_overlap_f1', 'gray_mae', 'shift_x_normalized_px',
              'shift_y_normalized_px', 'low_information_image', 'exact_sha256_match',
              'main_quality_status', 'mad_quality_status', 'main_image_path', 'mad_image_path',
              'main_alignment_status', 'main_extra_rendered_pages',
              'main_current_mapping_rows', 'mad_current_mapping_rows', 'review_status']
    files = []
    try:
        f, candidates = writer(args.output_dir, 'candidates.csv', common); files.append(f)
        f, summaries = writer(args.output_dir, 'summary.csv', ['stack_id', 'main_masterindex_id',
            'mad_masterindex_id', 'mad_pdf_page_number', 'main_alignment_status', 'main_mapping_rows',
            'main_inventory_pages', 'main_readable_pages', 'mad_inventory_pages', 'best_main_pdf_page_number',
            'best_phash_distance_63bits', 'candidates_within_cutoff', 'status']); files.append(f)
        f, errors = writer(args.output_dir, 'errors.csv', ['stack_id', 'masterindex_id', 'pdf_page_number',
                                                          'image_path', 'error']); files.append(f)
        f, skips = writer(args.output_dir, 'skipped_stacks.csv', ['stack_id', 'status']); files.append(f)
        skips.writerows(skipped)
        f, scope = writer(args.output_dir, 'mid_scope.csv', ['stack_id', 'masterindex_id',
            'mapping_rows', 'successful_rendered_pages', 'extra_rendered_pages',
            'mapping_alignment_status', 'decision']); files.append(f)
        scope.writerows(scope_rows)
        total = 0
        for index, (stack, refs, targets) in enumerate(jobs, 1):
            print(f'[{index}/{len(jobs)}] {stack}', flush=True)
            # Bound memory to one target MID at a time; reference pages reused.
            ref_pages = {mid: [load_page(r, args, args.rotations, errors, stack)
                               for r in reference_pages(groups[(stack, mid)], inventory.get(mid, []))] for mid in refs}
            for mid in targets:
                target_pages = [p for r in inventory.get(mid, [])
                                if (p := load_page(r, args, [0], errors, stack)) is not None]
                group = groups[(stack, mid)]
                status = '|'.join(sorted(group['statuses']))
                for refmid in refs:
                    if refmid == mid:
                        continue
                    refs_available = ref_pages[refmid]
                    # Empty references still produce an explicit summary row.
                    for ref in refs_available or [None]:
                        summary = {'stack_id': stack, 'main_masterindex_id': mid,
                            'mad_masterindex_id': refmid, 'mad_pdf_page_number': ref['source_page_number'] if ref else '',
                            'main_alignment_status': status, 'main_mapping_rows': len(group['rows']),
                            'main_inventory_pages': len(inventory.get(mid, [])), 'main_readable_pages': len(target_pages),
                            'mad_inventory_pages': len(inventory.get(refmid, []))}
                        if ref is None or not target_pages:
                            summary['status'] = 'MISSING_OR_UNREADABLE_REFERENCE' if ref is None else 'NO_READABLE_TARGET_PAGES'
                            summaries.writerow(summary)
                            continue
                        ranked = []
                        for target in target_pages:
                            distance, angle = min(((rh ^ target['features'][0][0]).bit_count(), angle)
                                for angle, (rh, _, _) in ref['features'].items())
                            ranked.append((distance, target['source_page_number'], angle, target))
                        ranked.sort(key=lambda item: item[:3])
                        summary.update(best_main_pdf_page_number=ranked[0][1], best_phash_distance_63bits=ranked[0][0],
                            candidates_within_cutoff=sum(d <= args.phash_distance for d, _, _, _ in ranked),
                            status='REVIEW_SUGGESTIONS_ONLY')
                        summaries.writerow(summary)
                        for rank, (distance, page, angle, target) in enumerate(ranked[:args.top_k], 1):
                            _, refgray, ref_low = ref['features'][angle]
                            _, targetgray, target_low = target['features'][0]
                            ink, mae, dx, dy = detail_scores(refgray, targetgray)
                            exact = bool(ref.get('image_sha256')) and ref.get('image_sha256') == target.get('image_sha256')
                            low = ref_low or target_low
                            review = 'LOW_INFORMATION_REVIEW' if low else ('EXACT_HASH_REVIEW' if exact else
                                ('VISUAL_CANDIDATE_REVIEW' if distance <= args.phash_distance else 'WEAK_CANDIDATE_REVIEW'))
                            candidates.writerow(dict(stack_id=stack, main_masterindex_id=mid, main_pdf_page_number=page,
                                mad_masterindex_id=refmid, mad_pdf_page_number=ref['source_page_number'], candidate_rank=rank,
                                phash_distance_63bits=distance, within_review_cutoff=distance <= args.phash_distance,
                                reference_rotation_degrees=angle, ink_overlap_f1=round(ink, 6), gray_mae=round(mae, 6),
                                shift_x_normalized_px=dx, shift_y_normalized_px=dy, low_information_image=low,
                                exact_sha256_match=exact, main_quality_status=target.get('quality_status', ''),
                                mad_quality_status=ref.get('quality_status', ''), main_image_path=target['resolved_image_path'],
                                mad_image_path=ref['resolved_image_path'], main_alignment_status=status,
                                main_extra_rendered_pages=len(inventory[mid]) - len(group['rows']),
                                main_current_mapping_rows=mapped_rows(group, page),
                                mad_current_mapping_rows=mapped_rows(groups[(stack, refmid)], ref['source_page_number']),
                                review_status=review))
                            total += 1
            for f in files:
                f.flush()
        run = {'mapping': str(args.mapping.resolve()), 'inventory': str(args.inventory.resolve()),
               'stacks_processed': len(jobs), 'stacks_skipped': len(skipped),
               'mapping_mid_groups': len(groups), 'candidate_rows': total, 'top_k': args.top_k,
               'target_scope': 'Inventory-confirmed extra rendered pages relative to mapping rows; CSV status is informational',
               'rendered_mapping_mids': len(inventory),
               'phash_review_cutoff': args.phash_distance, 'rotations': args.rotations,
               'requested_stacks_not_processed': sorted(selected - {j[0] for j in jobs}),
               'notice': 'Suggestions only. Scores are not probabilities. No automatic deletion or mapping edits. '
                         'Current mapping labels may be wrong. Cross-reference matches are not globally one-to-one. '
                         'References are pure MAD MIDs or MAD pages of aligned mixed MIDs. '
                         'No reference means no content-based MAD detection; see skipped_stacks.csv and summary.csv.'}
        with (args.output_dir / 'run_info.json').open('x', encoding='utf-8') as f:
            json.dump(run, f, indent=2, ensure_ascii=False)
        print(f'Saved {total} candidate rows: {args.output_dir.resolve()}')
    finally:
        for f in files:
            f.close()


if __name__ == '__main__':
    main()
