import sys
import logging
import pickle
import pandas as pd
from data.prepare_data import prepare_data

logging.basicConfig(filename='/home/skye/Documents/Programming/sample-sklearn-ml-workflow.log',
                    filemode='w',
                    format='%(asctime)s: %(name)s - %(levelname)s - %(message)s')

def get_predictions( data_path="", model_path=""):
    champion_path = "{0}{1}".format(model_path, 'champion.pkl')
    print("0 "+champion_path)
    champion = pickle.load(open(champion_path, 'rb'))

    # Future extension: Data path or data retrieval process for consistently updated data
    _, _, X_test, test_id = prepare_data(data_path)

    # Future extension: Write to another file, database, etc.
    print(champion.predict(X_test))
    submission_path = "{0}{1}".format(model_path, "submission.csv")
    print(submission_path)

    submission = pd.DataFrame({'PassengerId': test_id, 'Survived': champion.predict(X_test)})
    submission.to_csv(submission_path)

if __name__ == '__main__':
    print('Version is', sys.version)
    print('sys.argv is', sys.argv)

    get_predictions("/home/skye/Documents/Programming/sample-sklearn-ml-workflow/data/",
                    "/home/skye/Documents/Programming/sample-sklearn-ml-workflow/output/")


    #try:
    #    get_predictions(sys.argv[1], sys.argv[2])
    #except:
    #    get_predictions()