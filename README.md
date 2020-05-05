# ALASKA2 Image Steganalysis from Kaggle

Repository for the SlimBros team for the [ALASKA2 Image Steganalysis challenge](https://www.kaggle.com/c/alaska2-image-steganalysis/) from Kaggle (May 2020 to July 2020). In this competition, In the competition, the goal is to create an efficient and reliable method to detect secret data hidden within innocuous-seeming digital images. 

## Table of contents
  * [Team](#team)
  * [Timeline](#timeline)
  * [File structure](#file-structure) 
  * [Models](#models)
  * [Current status](#current-status) 
  * [Useful links](#useful-links)

## Team
- [Guillem Delgado](https://www.linkedin.com/in/guillem-delgado-gonzalo-576aa73a/)

## Timeline
- July 13, 2020 - Entry deadline. Must accept the competition rules before this date in order to compete.
- July 13, 2020 - Team Merger deadline. This is the last day participants may join or merge teams.
- July 20, 2020 - Final submission deadline.

All deadlines are at 11:59 PM UTC on the corresponding day unless otherwise noted. The competition organizers reserve the right to update the contest timeline if they deem it necessary.

## File structure
```
.
|-- 1. README.md
|-- 2. training.py
|-- data_loader
|   `-- generator.py
|-- model
|   `-- regression.py
|-- utils
|   |-- metrics.py
|   `-- utils.py

```

## Models
- [] Regression
  - [ ] ensenet
- [ ] GAN
  - [ ] State of the Art for Steganalysis 
- [ ] Stacking the classical ML models

## Ongoing tasks
- [ ] Data augmentation
  - [ ] Generate shifts
- [ ] Learning rate Scheduler

## Useful links
- [Challenge website](https://www.kaggle.com/c/alaska2-image-steganalysis/)
- [Submit predictions](https://www.kaggle.com/c/alaska2-image-steganalysis/submit)
- [Leaderboard](https://www.kaggle.com/c/alaska2-image-steganalysis/leaderboard)
