# Machine Learning Project II

This is a project for the EPFL class Machine Learning CS-433 in collaboration with the ESA lab ESTEC.

In this project, we will predict coronal mass ejections with the data from EDAC counters and magnetic sensors of several spacecrafts.

This project contains our final report to explain everything in it.


## Agatha DUZAN | Sebastien CHAHOUD | Nastasia MOREL

- [@agatha-duzan](https://www.github.com/agatha-duzan)
- [@Sebibi](https://www.github.com/Sebibi)
- [@nastasia1234](https://www.github.com/nastasia1234)


## Installation

1 - Clone the repo

```bash
  git clone https://github.com/CS-433/ml-project-2-ml4esa
```

2 - Add the data

Some data files were too large to be on the repo, you can find them on this drive : 
https://drive.google.com/drive/folders/1ZhoJT79JUiJTjR9oS9j5ZufEsILzMR2O?usp=sharing

Download the data files and add them to the DATA folder.

## Structure of the repo

- **Data fetching and preprocessing**: all the code we used to fetch the data online and preprocess it can be found here. It also contains the first visualizations of the data.
- **DATA** folder: contains the original data, scraped data and cleaned up data. It allows you to run the models without having to reproduce the scraping and preprocessing.
- **main** notebook: Training and evaluation of our final model.
- **Other notebooks**: other models that we experimented with (different architectures, input features and hyperparameters)
- **Final report**: contains the project report which describes in detail the project, our approach, the different models, the results, and the ethical considerations.