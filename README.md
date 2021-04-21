# Turath-150K Database

This repository contains the data for three distinct benchmark databases and
code for training and evaluating supervised neural networks on the task of 
image classification. 

Turath, in Arabic, roughly means heritage. Broadly, the database consists of objects, activities, and scenarios 
commonly encountered in the Arab World (from Mauritania in the West of Africa to Iraq). More specifically, there 
exist 3 distinct benchmark databases; Turath-Standard, Turath-Art, and Turath-UNESCO, whose contents are described
next. 

## Turath Benchmark Databases

### Turath Standard

This benchmark comprises images reflecting categories with low-level details (e.g., food and clothing) to those 
with high-level details (e.g., cities and architecture). It has a total of 12 macro, image-level categories, each of
which consist of more granular micro categories, with a total of Y. On average, each micro category contains between
50 and 500 images.

### Turath Art

This benchmark comprises images reflecting art pieces produced by Arab artists. It has a total of 391 image-level
categories with, on average, 50-500 images per category. The list of names of these artists, although far from
exhaustive, was obtained from the [Barjeel Art Foundation](https://www.barjeelartfoundation.org/).

### Turath UNESCO

This benchmark comprises images reflecting the UNESCO heritage sites in the Arab World. It has a total of 79 image-level categories with, on average, 50-500 images per category. 

## Training Networks

To reproduce the results of the experiments originally conducted on the three distinct benchmark databases, please 
follow these instructions:

1. Download the Turath database found [here]()
2. Open the Google Collab [notebook]() and follow the instructions there 

## Citing Turath-150K

If the Turath-150K database has inspired you or facilitated your research, please consider citing the following paper:
