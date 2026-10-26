import os

import numpy as np
import pandas as pd
import torch
from pnc_claims_classification.training.config import (
    SCHADEN_DATASET_SUFFIX,
    SHARED_FOLDER_ROOT_DIR,
)

default_data_file = f"usecases/docai/classifile/meta/schaden_202324/schaden_202324_page_level_meta_file_renamed_{SCHADEN_DATASET_SUFFIX}_mapped.parquet"


def prepare_dataset(
    data_file: str = default_data_file,
    id_to_class: dict[int, str] | None = None,
) -> tuple[
    dict[str, dict[str, np.typing.NDArray]],
    dict[int, str],
    dict[str, int],
    dict[int, str],
    np.typing.NDArray,
]:
    """
    Prepare dataset for training by loading data and creating class mappings.
    This function loads a parquet dataset, adds mappings between class IDs and class names,
    and prepares the training data structure. It handles class enumeration automatically
    if no explicit mapping is provided.
    Args:
        data_file (str, optional): Path to the data file relative to SHARED_FOLDER_ROOT_DIR.
            Defaults to default_data_file.
        id_to_class (dict[int, str] | None, optional): Mapping from class IDs to class names.
            If None, classes will be automatically enumerated from sorted unique target values.
            Defaults to None.
    Returns:
        tuple: A 5-element tuple containing:
            - dataset (dict[str, dict[str, np.typing.NDArray]]): Dictionary with 'train' key
              containing the loaded training data
            - id_to_class (dict[int, str]): Mapping from class IDs to class names
            - class_to_id (dict[str, int]): Mapping from class names to class IDs
            - id_remapper_names (dict[int, str]): Copy of id_to_class mapping for remapping
            - id_remapper (np.typing.NDArray): Array for ID remapping operations
    Note:
        The function expects the parquet file to have a 'target' column containing
        the class labels.
    """

    meta = pd.read_parquet(os.path.join(SHARED_FOLDER_ROOT_DIR, data_file))

    print(f"Loaded dataframe with {len(meta)} rows")

    if not id_to_class:
        id_to_class = {
            id: cls for id, cls in enumerate(sorted(meta.target.unique().tolist()))
        }

    class_to_id = {cls: id for id, cls in id_to_class.items()}

    id_remapper = np.arange(len(id_to_class))

    # Free up memory
    del meta

    print("Dataset with ", len(id_to_class), "classes")

    id_remapper_names = id_to_class

    dataset = {
        "train": load_data(
            "train",
            data_file,
            class_to_id,
        ),
    }

    return dataset, id_to_class, class_to_id, id_remapper_names, id_remapper


def build_validation_set(
    dataset: dict[str, dict[str, np.typing.NDArray]], split_ratio: float = 0.9
) -> dict[str, dict[str, np.typing.NDArray]]:
    """
    Build a validation set by splitting the training data.
    Takes the existing training data and splits it into new training and validation sets
    based on the specified split ratio. The split is performed at the stack level using
    a fixed random seed for reproducibility.
    Args:
        dataset: Dictionary containing dataset splits with 'train' key containing
                training data arrays including stacks, features, text, classes, etc.
        split_ratio: Float between 0 and 1 indicating the proportion of data to keep
                    for training. Default is 0.9 (90% train, 10% validation).
    Returns:
        Dictionary with the same structure as input but with updated 'train' split
        and new 'val' split. The original training data is split between these two.
    Note:
        - Uses a fixed random seed (5) for reproducible splits
        - Splits are performed at the stack level to maintain data integrity
        - Both new splits are flattened after creation
        - Validates that total number of pages is preserved after split
    """

    print("Building validation set from training set...")
    train_split = dataset["train"]
    new_train_split = {}
    val_split = {}

    rng = np.random.RandomState(5)
    stacks_perm = rng.permutation(len(train_split["stacks"]))
    print("Using split ratio", split_ratio)
    split = int(split_ratio * len(stacks_perm))
    stacks_train = sorted(stacks_perm[:split])
    stacks_val = sorted(stacks_perm[split:])

    for k in [
        "stacks",
        "stacks_isfirstpage",
        "stacks_islastpage",
        "stacks_pagenum_beg",
        "stacks_pagenum_end",
        "stacks_features",
        "stacks_text",
        "stacks_classes",
    ]:
        if k not in train_split:
            continue
        val_split[k] = train_split[k][stacks_val]
        new_train_split[k] = train_split[k][stacks_train]

    val_split = flatten_stacks(val_split)
    new_train_split = flatten_stacks(new_train_split)

    dataset["val"] = val_split
    dataset["train"] = new_train_split

    # Check consistency
    assert len(dataset["train"]["pages_stacks"]) + len(
        dataset["val"]["pages_stacks"]
    ) == len(train_split["pages_stacks"])

    print(
        "Train/val stacks:",
        len(dataset["train"]["stacks"]),
        len(dataset["val"]["stacks"]),
    )
    return dataset


def flatten_stacks(
    stacks_dict: dict[str, np.typing.NDArray],
) -> dict[str, np.typing.NDArray]:
    """
    Flatten nested stacks of documents and pages into different representations.
    Takes a dictionary containing stacks of documents (where each document contains pages)
    and creates flattened versions with different levels of granularity.
    Args:
        stacks_dict (dict[str, np.typing.NDArray]): Dictionary where keys are stack names
            and values are arrays of stacks. Each stack contains documents, and each
            document contains pages.
    Returns:
        dict[str, np.typing.NDArray]: Updated dictionary with original data plus new keys:
            - "flattened_{k}": Flattened stacks where each stack becomes a single list of pages
            - "docs_{k}": All documents from all stacks concatenated into one array
            - "pages_{k}": All pages from all documents concatenated, with concatenation
              method depending on data type (torch.cat for tensors, np.concatenate otherwise)
    Note:
        The function modifies the input dictionary in-place and returns it.
        Handles different data types appropriately:
        - torch.Tensor: Uses torch.cat for concatenation
        - String arrays: Uses np.concatenate with object dtype
        - Other numeric arrays: Uses standard np.concatenate
    """

    for k, stacks in list(stacks_dict.items()):
        flattened = []
        docs = []
        for stack in stacks:
            docs += stack
            current_stack_pages = []
            for doc in stack:
                current_stack_pages += doc
            flattened.append(current_stack_pages)
        stacks_dict[f"flattened_{k}"] = np.array(flattened, dtype=object)
        stacks_dict[f"docs_{k}"] = np.array(docs, dtype=object)

        if isinstance(docs[0], torch.Tensor):
            stacks_dict[f"pages_{k}"] = torch.cat(docs, dim=0)
        elif isinstance(docs[0][0], str):
            stacks_dict[f"pages_{k}"] = np.concatenate(docs, axis=0, dtype=object)
        else:
            stacks_dict[f"pages_{k}"] = np.concatenate(docs, axis=0)

    return stacks_dict


def load_data(
    split: str,
    data_file: str,
    class_to_id: dict[str, int],
) -> dict[str, np.typing.NDArray]:
    """
    Load and preprocess data from a parquet file for document classification.
    This function loads document metadata from a parquet file, processes it to create
    document boundaries and page information, filters by valid classes, and returns
    a flattened dataset structure.
    Args:
        split: The data split identifier (e.g., 'train', 'val', 'test')
        data_file (str): Path to the parquet file containing the data
        class_to_id (dict[str, int]): Mapping of class names to integer IDs for filtering
    Returns:
        The flattened dataset output containing processed document stacks
    Notes:
        - Creates document IDs based on cumulative sum of first page flags
        - Adds last page indicators and merges page count statistics
        - Filters data to only include valid classes defined in class_to_id
        - Sorts data by stack_id and image_id before processing
        - Prints warnings if records are dropped due to invalid classes
    """

    meta = pd.read_parquet(os.path.join(SHARED_FOLDER_ROOT_DIR, data_file))
    meta["split"] = split
    meta["doc_id"] = meta["doc_first_page"].cumsum().apply(str)
    lastpage_col = meta["doc_first_page"].values[1:].tolist()
    lastpage_col.append(True)  # type: ignore[union-attr, arg-type]
    meta["act_lastpage"] = lastpage_col

    meta["sfdoc_class"] = meta["target"]
    del meta["target"]

    meta["act_firstpage"] = meta["doc_first_page"]
    if split == "train":
        assert meta["act_firstpage"].sum() == meta["act_lastpage"].sum(), (
            meta["act_firstpage"].sum(),
            meta["act_lastpage"].sum(),
        )
    counts = meta.groupby("stack_id")["doc_id"].count()
    counts.name = "total_n_pages"
    meta = meta.merge(counts, left_on="stack_id", right_index=True)

    pre_len = len(meta)
    meta = meta[meta.sfdoc_class.isin(class_to_id.keys())]
    post_len = len(meta)
    if pre_len != post_len:
        print(
            f"Warning in split {split}: dropped records ({pre_len} -> {post_len}) due to invalid classes"
        )

    meta_sorted = meta.sort_values(["stack_id", "image_id"])

    stacks_dict = create_stacks_dict(
        meta_sorted=meta_sorted,
        split=split,
        class_to_id=class_to_id,
    )

    output = flatten_stacks(stacks_dict)

    print("Loaded", split)
    return output


def create_stacks_dict(
    meta_sorted: pd.DataFrame,
    split: str,
    class_to_id: dict[str, int],
) -> dict[str, np.typing.NDArray]:
    """Create a dictionary containing organized data structures for model
    training/evaluation.

    This function processes a sorted metadata DataFrame and organizes the data into stacks,
    where each stack contains multiple documents, and each document contains multiple pages.
    The function tracks first page and last page labels, text content, and document classes.

    Args:
        meta_sorted (pd.DataFrame): Sorted DataFrame containing metadata with columns:
            - doc_id: Document identifier
            - stack_id: Stack identifier grouping related documents
            - act_firstpage: Boolean indicating if this is the first page of a document
            - act_lastpage: Boolean indicating if this is the last page of a document
            - filename: Path to the document file
            - text: Text content of the page
            - sfdoc_class: Document classification label
        split (str): Split type identifier. When "eval", ensures each document has at least
            one first page by overriding flags if necessary.
        class_to_id (dict[str, int]): Mapping from class names to class IDs.

    Returns:
        dict[str, np.typing.NDArray]: Dictionary containing organized data with keys:
            - "stacks": Array of stacks, where each stack contains documents (lists of file paths)
            - "stacks_isfirstpage": Array of first page labels for each page in each document
            - "stacks_islastpage": Array of last page labels for each page in each document
            - "stacks_text": Array of text content for each page in each document
            - "stacks_classes": Array of class IDs for each page in each document

    Note:
        - Documents are grouped by stack_id and doc_id
        - For evaluation splits, first page flags may be automatically corrected
        - Unknown classes are assigned ID -100
        - File paths are adjusted for documents starting with "doc" prefix
    """

    stacks = []
    labels_firstpage = []
    labels_lastpage = []
    texts = []
    classes = []
    current_stack = None
    current_labels_firstpage_stack = None
    current_labels_lastpage_stack = None
    current_text_stack = None
    current_class_stack = None
    current_doc = None
    current_labels_firstpage_doc = None
    current_labels_lastpage_doc = None
    current_text_doc = None
    current_class_doc = None
    current_stack_id = None
    current_doc_id = None  # For verification purposes
    cnt_overridden = 0
    for _, row in meta_sorted.iterrows():
        if (
            current_doc_id != row["doc_id"]
            and split == "eval"
            and not row["act_firstpage"]
        ):
            cnt_overridden += 1
            row["act_firstpage"] = True
        if row["act_firstpage"]:
            if current_doc is not None and len(current_doc) > 0:
                current_stack.append(current_doc)
                current_labels_firstpage_stack.append(current_labels_firstpage_doc)
                current_labels_lastpage_stack.append(current_labels_lastpage_doc)
                current_text_stack.append(current_text_doc)
                current_class_stack.append(current_class_doc)
            current_doc = []
            current_labels_firstpage_doc = []
            current_labels_lastpage_doc = []
            current_text_doc = []
            current_class_doc = []
            assert current_doc_id != row["doc_id"]
            current_doc_id = row["doc_id"]
        else:
            assert current_doc_id == row["doc_id"]
        if current_stack_id != row["stack_id"]:
            if current_stack is not None and len(current_stack) > 0:
                stacks.append(current_stack)
                labels_firstpage.append(current_labels_firstpage_stack)
                labels_lastpage.append(current_labels_lastpage_stack)
                texts.append(current_text_stack)
                classes.append(current_class_stack)
            current_stack = []
            current_labels_firstpage_stack = []
            current_labels_lastpage_stack = []
            current_text_stack = []
            current_class_stack = []
        current_stack_id = row["stack_id"]
        path = (
            row["filename"]
            if not row["filename"].startswith("doc")
            else "/tmp/s3/ap-mlops-pnc-claims-docai/" + row["filename"]  # nosec
        )
        current_doc.append(path)
        current_labels_firstpage_doc.append(1 if row["act_firstpage"] else 0)
        current_labels_lastpage_doc.append(1 if row["act_lastpage"] else 0)
        current_text_doc.append(row["text"].strip())
        current_class_doc.append(
            class_to_id[row["sfdoc_class"]]
            if row["sfdoc_class"] in class_to_id
            else -100
        )
    if current_doc is not None and len(current_doc) > 0:
        current_stack.append(current_doc)
        current_labels_firstpage_stack.append(current_labels_firstpage_doc)
        current_labels_lastpage_stack.append(current_labels_lastpage_doc)
        current_text_stack.append(current_text_doc)
        current_class_stack.append(current_class_doc)
    if len(current_stack) > 0:
        stacks.append(current_stack)
        labels_firstpage.append(current_labels_firstpage_stack)
        labels_lastpage.append(current_labels_lastpage_stack)
        texts.append(current_text_stack)
        classes.append(current_class_stack)

    if cnt_overridden != 0:
        print("Some firstpage flags were realigned. Count =", cnt_overridden)

    stacks_dict = {
        "stacks": np.array(stacks, dtype=object),
        "stacks_isfirstpage": np.array(labels_firstpage, dtype=object),
        "stacks_islastpage": np.array(labels_lastpage, dtype=object),
        "stacks_text": np.array(texts, dtype=object),
        "stacks_classes": np.array(classes, dtype=object),
    }

    return stacks_dict
