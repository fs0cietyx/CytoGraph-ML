# Zenodo Submission Guide for CytoGraph-ML

When uploading the **CytoGraph-ML** preprint/manuscript and codebase to Zenodo, follow these specific guidelines to properly link your repository and datasets.

## 1. Related Works Section

The **Related works** section on Zenodo is strictly for linking your submission to materials you built upon or external code/data dependencies. Use the following configuration:

### Entry 1: Linking the GitHub Code Repository
This tells Zenodo that your paper is supplemented by an open-source software implementation.
* **Identifier:** `https://github.com/fs0cietyx/CytoGraph-ML`
* **Scheme:** `URL`
* **Relation:** `is supplemented by` (or *is documented by*)
* **Resource type:** `Software`

### Entry 2: Linking the Foundational Dataset (GSE10072)
This gives credit to the core clinical dataset that trained your pipeline.
* **Identifier:** `https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE10072`
* **Scheme:** `URL`
* **Relation:** `is derived from`
* **Resource type:** `Dataset`

### Entry 3: Linking the External Validation Dataset (GSE19804)
* **Identifier:** `https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE19804`
* **Scheme:** `URL`
* **Relation:** `is derived from`
* **Resource type:** `Dataset`

### Entry 4: Linking the MyGene.info API (BioMapper Dependency)
This attributes the external service used for biological pathway enrichment.
* **Identifier:** `10.1186/s13059-016-0953-9`
* **Scheme:** `DOI`
* **Relation:** `requires`
* **Resource type:** `Publication` (or Software)

## 2. References Section

**Important:** Do not confuse the "Related works" section with the "References" text box.

* **Related works:** Only for items with a direct interactive or structural link to your code/data (requiring a DOI or URL).
* **References:** This is where you paste your standard text-based bibliography (the list of papers cited in your manuscript's literature review).
