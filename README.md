# ALASKA2 Image Steganalysis from Kaggle

Repository for the SlimBros team for the [ALASKA2 Image Steganalysis challenge](https://www.kaggle.com/c/alaska2-image-steganalysis/) from Kaggle (May 2020 to July 2020). In this competition, In the competition, the goal is to create an efficient and reliable method to detect secret data hidden within innocuous-seeming digital images. 

## Table of contents
  * [Team](#team)
  * [Timeline](#timeline)
  * [File structure](#file-structure) 
  * [Models](#models)
  * [Current status](#current-status) 
  * [Useful links](#useful-links)
  * [Contributing](#contributing)
  * [License](#License)

## Team
- [Guillem Delgado](https://www.linkedin.com/in/guillemdelgado/)
- [Ricard Delgado](https://www.linkedin.com/in/ricarddelgadogonzalo/)

## Timeline
- July 13, 2020 - Entry deadline. Must accept the competition rules before this date in order to compete.
- July 13, 2020 - Team Merger deadline. This is the last day participants may join or merge teams.
- July 20, 2020 - Final submission deadline.

All deadlines are at 11:59 PM UTC on the corresponding day unless otherwise noted. The competition organizers reserve the right to update the contest timeline if they deem it necessary.

## File structure
```
.
|-- 1. README.md
|-- pytorch
|   `-- config
|   `-- data_loader
|   `-- model
|   `-- trainer
|   `-- utils
|   `-- evaluator.py
|   `-- main.py
|   `-- training.py (deprecated)
|-- tensorflow
|   `-- data_loader
|   `-- model
|   `-- training.py
|   `-- blending.py
|-- utils
|   |-- metrics.py
|   `-- utils.py

```

## Models
- [x] Regression
  - [x] Rensenet
- [x] Classification (4/9 classes)
  - [x] EfficientNet
  - [x] Visual Attention

## Usage
Launch pytorch/main.py file for the up-to-date changes and the best resulting score. The evaluator.py file will generate a submition for the test set. Modify accordingly the JSON file inside the config folder in order to use the scripts. 

## Useful links
- [Challenge website](https://www.kaggle.com/c/alaska2-image-steganalysis/)
- [Submit predictions](https://www.kaggle.com/c/alaska2-image-steganalysis/submit)
- [Leaderboard](https://www.kaggle.com/c/alaska2-image-steganalysis/leaderboard)

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.

## License
[MIT](https://choosealicense.com/licenses/mit/)
