<div align="center">

# Hyperbolic Contrastive Learning with Entailment for Spatial Transcriptomics

**Daniela Vega\*, Paula Cárdenas, Hannah Ceballos, Leonardo Manrique, and Pablo Arbeláez**

Universidad de los Andes, Colombia

</div>

## News

- **2026:** HyCLoST was accepted to **MICCAI 2026**.

## Abstract

Spatial Transcriptomics (ST) has transformed biomedical research by enabling the spatial mapping of gene expression across tissue sections. However, high operational costs, specialized equipment requirements, and sensitivity to experimental noise limit the accessibility and scalability of ST. Recent computer vision approaches aim to overcome these limitations by predicting spatial gene expression directly from histopathology images. While effective, current approaches often suffer from gene expression over-smoothing and overly uniform predictions across tissue regions, suggesting that further progress depends on learning representations that reflect the hierarchical and asymmetric structure of gene regulation and tissue morphology. To address these issues, we propose Hyperbolic Contrastive Learning with Entailment for Spatial Transcriptomics (HyCLoST), a hyperbolic contrastive learning model that captures the intrinsic hierarchical relationships within ST data. By leveraging hyperbolic geometry and a gene-to-image entailment loss, HyCLoST learns structured, biologically grounded representations that improve gene expression prediction accuracy, achieving a 6% reduction in MSE and an 8% increase in PCC across 26 ST datasets, over previous methods.

## Main Figure

![HyCLoST overview](complete_plot_v3.png)

**Figure 1.** (a) HyCLoST learns a joint hierarchical representation of histopathology patches and corresponding gene expression vectors via contrastive learning in hyperbolic space. (b) During training, image patches and paired gene expression vectors are encoded with UNI2-h and Geneformer and projected into a shared hyperbolic space. (c) A patch-gene entailment objective models the asymmetric relationship between modalities. (d) In inference, a patch is encoded and projected into the learned hyperbolic space, and its prediction is obtained by averaging the *k* nearest gene embeddings.

## Citation

```bibtex
@inproceedings{vega2026hyclost,
  title     = {Hyperbolic Contrastive Learning with Entailment for Spatial Transcriptomics},
  author    = {Vega, Daniela and Cárdenas, Paula and Ceballos, Hannah and Manrique, Leonardo and Arbeláez, Pablo},
  booktitle = {International Conference on Medical Image Computing and Computer-Assisted Intervention},
  year      = {2026}
}
```
