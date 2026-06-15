# External Cohort Validation Protocols

To evaluate real-world clinical feasibility, models must survive validation on entirely unseen datasets collected using different sequencing platforms and equipment.

## 1. Cross-Platform Validation (GSE10072 -> GSE19804)

- **Source Dataset (GSE10072):** Analyzed on the Affymetrix GPL96 platform.
- **Target Dataset (GSE19804):** Analyzed on the Affymetrix GPL570 platform.

This platform shift introduces massive batch effects. CytoGraph-ML addresses this shift using rank-based `QuantileNormalizer` alignment.

## 2. Cross-Tissue Transfer Validation (GSE10072 -> GSE21510)

- **Source Tissue (GSE10072):** Lung cancer vs normal lung tissue.
- **Target Tissue (GSE21510):** Colorectal cancer vs normal colorectal tissue.

This test evaluates whether the identified biomarkers represent a universal oncogenic pathway (e.g. angiogenesis, hypoxia, or cell junction degradation) that transfers across different epithelial cancers.

## 3. Dynamic Feature Schema Alignment

When verifying models on external datasets:
1. The overlapping features (Gene Symbols) between training and validation datasets are computed.
2. The dataset matrices are re-indexed to this common subset.
3. Missing genes are safely padded with zeros to ensure model consistency.
