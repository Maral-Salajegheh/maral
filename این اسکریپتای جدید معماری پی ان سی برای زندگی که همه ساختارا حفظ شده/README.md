# Life DocAI

Life DocAI contains analytics and document-processing components for Life insurance documents.

The current implementation focuses on the **Antrag** workflow and provides reusable components for preparing document data, mapping pages and documents, classifying document pages, and producing downstream results for further processing.

## Repository structure

The main implementation is located under:

```text
packages/life-docai-analytics/
```

The Antrag-specific package is:

```text
life_docai_analytics_antrag/
```

It currently contains the following main components:

- **datasets/**  
  Data preparation, ingestion, and mapping utilities used to create the inputs required by downstream processing.

- **classification/**  
  Page classification and related processing, including training, evaluation, and additional document-specific logic.

Each major component contains its own README with detailed setup, inputs, outputs, and execution instructions.

## Environment

The repository uses **Pixi** for environment and dependency management.

The relevant project configuration is defined in:

```text
pixi.toml
pixi.lock
pyproject.toml
```

## Workflow

At a high level, the current Antrag workflow is:

```text
Data ingestion
      ↓
Document / page mapping
      ↓
Classification
      ↓
Evaluation and downstream processing
```

Generated datasets, intermediate outputs, model artifacts, and local runtime files are not stored in Git and are excluded through `.gitignore`.

## Documentation

Detailed documentation for each processing stage is available in the corresponding package directories.
