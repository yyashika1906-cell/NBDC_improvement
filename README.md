# NBDC_Public

Instead of web scraping, I accessed the GBIF API to extract image URLs for each species

## Workflow
For GBIF's API, I had to create a URL and fill the search parameters to get results that contained image URLs for each species, somewhat similarly to web scraping.

Here is an example of such URL so you can see the result:
https://api.gbif.org/v1/occurrence/search?scientificName=Andrena+miserabilis&mediaType=StillImage&limit=50&offset=350&basisOfRecord=PRESERVED_SPECIMEN&license=CC0_1_0&taxonKey=7798
- Using this, I would change the parameters `scientificName` according to the species I was searching for, and I would increment `offset` by 50 to get the next 50 results (each page had 50 results since `limit=50`).

I would run `main.py` to get the image URLs for each species, which would be saved in the `species_images` folder.
I would then run `download.py` to get these images and save them to the correct species folder, as usual.

## Other notes

### apiUtils.py
`excluded_dataset_keys` exists because we were told to exclude the 'iNaturalist' and 'BugGuide' from our search, as they include civilian observations which may not correctly label bees.

Some species had no images, and for those species I searched for their names online to look for *synonyms*, which are slightly different names for the same species of bee.
- If API had images for a synonym, I would add it to the `synonyms` dictionary.
  - Make sure the folder you are saving the images in follows the original name, **not** the synonym name
- Some did not have images for a synonym of a species, so they were left out.

### count.py
I used this file to create a copy of `bee_counts.csv`, where the GBIF column has the updated counts for each species according to the results I downloaded.