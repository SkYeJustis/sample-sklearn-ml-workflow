from models.gradboost import create_best_gradboost
from models.logreg import create_best_logreg
from models.randomforest import create_best_rf
import logging
import sys
import numpy as np
import pickle

logging.basicConfig(filename='/home/skye/Documents/Programming/sample-sklearn-ml-workflow.log',
                    filemode='w',
                    format='%(asctime)s: %(name)s - %(levelname)s - %(message)s')

def train_champion_main(data_path="", output_path=""):
    paths = []
    assess_metric = []

    mod_1_path, mod_1_metric = create_best_gradboost(data_path, output_path)
    paths.append(mod_1_path)
    assess_metric.append(mod_1_metric)

    mod_2_path, mod_2_metric = create_best_rf(data_path, output_path)
    paths.append(mod_2_path)
    assess_metric.append(mod_2_metric)

    mod_3_path, mod_3_metric = create_best_logreg(data_path, output_path)
    paths.append(mod_3_path)
    assess_metric.append(mod_3_metric)


    # AUC is currently set as the assess metric
    champion_path = paths[np.argmax(np.array(assess_metric))]
    #logging.INFO("CHAMPION is {0} with a AUC score of {1}".format(champion_path, np.argmax(np.array(assess_metric))))
    champion = pickle.load(open(champion_path, 'rb'))

    model_path = "{0}{1}".format(output_path, 'champion.pkl')
    with open(model_path, 'wb') as file:
        pickle.dump(champion, file=file)



if __name__ == '__main__':
    print('Version is', sys.version)
    print('sys.argv is', sys.argv)

    train_champion_main("/home/skye/Documents/Programming/sample-sklearn-ml-workflow/data/",
                        "/home/skye/Documents/Programming/sample-sklearn-ml-workflow/output/")

    #try:
    #    train_champion_main(sys.argv[1])
    #except:
    #    train_champion_main("/home/skye/Documents/Programming/sample-sklearn-ml-workflow/data/")
