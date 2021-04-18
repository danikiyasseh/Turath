## Motivation

Large-scale image databases remain largely biased towards objects and activities encountered in a select few cultures. This absence of culturally-diverse images, which we refer to as the 'hidden tail', limits the applicability of pre-trained neural networks and inadvertently excludes researchers from under-represented regions. 

To begin remedying this issue, we curate Turath-150K, a database of images of Arab heritage that reflects objects, activities, and scenarios commonly found in the Arab world. As a consequence of Turath, we hope to engage machine learning researchers in under-represented regions, and to inspire the release of additional culture-focused databases. 

## Description

Turath, in Arabic, roughly means heritage. Broadly, the database consists of objects, activities, and scenarios 
commonly encountered in the Arab World (from Mauritania in the West of Africa to Iraq). More specifically, there 
exist 3 distinct benchmark databases; Turath-Standard, Turath-Art, and Turath-UNESCO, whose contents are described
next. 

### Turath Standard

This benchmark comprises images reflecting categories with low-level details (e.g., food and clothing) to those 
with high-level details (e.g., cities and architecture). It has a total of 12 macro, image-level categories, each of
which consist of more granular micro categories, with a total of 419. On average, each micro category contains between
50 and 500 images.

### Turath Art

This benchmark comprises images reflecting art pieces produced by Arab artists. It has a total of 391 image-level
categories with, on average, 50-500 images per category. The list of names of these artists, although far from
exhaustive, was obtained from the [Barjeel Art Foundation](https://www.barjeelartfoundation.org/).

### Turath UNESCO

This benchmark comprises images reflecting the UNESCO heritage sites in the Arab World. It has a total of 79 image-level categories with, on average, 50-500 images per category. 

## Demo

We also have several trained neural networks that you can use to directly perform inference on custom images. Please visit the following pages to see a demo of a neural network trained on [Turath-Standard](https://danikiyasseh.github.io/Turath/StandardDemo/), [Turath-Art](https://danikiyasseh.github.io/Turath/ArtDemo/), or [Turath-UNESCO](https://danikiyasseh.github.io/Turath/UNESCODemo/).
