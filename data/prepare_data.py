import pandas as pd
import logging

logging.basicConfig(filename='/home/skye/Documents/Programming/sample-sklearn-ml-workflow.log',
                    filemode='w',
                    format='%(asctime)s: %(name)s - %(levelname)s - %(message)s')

def prepare_data(data_path):

    # Edit this to switch data sources
    ## Currently: Using csv
    ## Can substitute with rest-api

    train = pd.read_csv('{0}train.csv'.format(data_path))
    test = pd.read_csv('{0}test.csv'.format(data_path))
    test_id = test['PassengerId']

    train = train.drop(['PassengerId', 'Cabin', 'Name', 'Ticket'], axis=1)
    test = test.drop(['PassengerId', 'Name', 'Cabin', 'Ticket'], axis=1)

    train['Age'].fillna(train['Age'].median(), inplace=True)
    test['Age'].fillna(test['Age'].median(), inplace=True)

    train['Embarked'].fillna(train['Embarked'].mode()[0], inplace=True)
    train['Embarked'].fillna(test['Embarked'].mode()[0], inplace=True)

    train['Fare'].fillna(train['Fare'].mode()[0], inplace=True)
    test['Fare'].fillna(test['Fare'].mode()[0], inplace=True)

    train = pd.concat([train, pd.get_dummies(train['Sex'], prefix='sex')], axis=1, sort=False)
    test = pd.concat([test, pd.get_dummies(test['Sex'], prefix='sex')], axis=1, sort=False)

    train = pd.concat([train, pd.get_dummies(train['Embarked'], prefix='Embarked')], axis=1, sort=False)
    test = pd.concat([test, pd.get_dummies(test['Embarked'], prefix='Embarked')], axis=1, sort=False)

    train = pd.concat([train, pd.get_dummies(train['Pclass'], prefix='Pclass')], axis=1, sort=False)
    test = pd.concat([test, pd.get_dummies(test['Pclass'], prefix='Pclass')], axis=1, sort=False)

    train = train.drop(['Sex', 'Embarked', 'Pclass'], axis=1)
    test = test.drop(['Sex', 'Embarked', 'Pclass'], axis=1)

    y = train['Survived']
    X = train.drop(['Survived'], axis=1)
    X_test = test

    return X, y, X_test, test_id