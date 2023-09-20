from .adaboost import Adaboost
from .decision_tree import RegressionTree, ClassificationTree, XGBoostRegressionTree
from .gradient_boosting import GradientBoostingClassifier, GradientBoostingRegressor
from .k_nearest_neighbors import KNN
from .regression import (
    LinearRegressor,
    LassoRegressor,
    RidgeRegressor,
    ElasticNetRegressor,
    PolynomialRegressor,
)
from .logistic_regression import LogisticRegression
from .multi_class_lda import MultiClassLDA
from .naive_bayes import NaiveBayes
from .random_forest import RandomForest
from .support_vector_machine import SupportVectorMachine
from .xgboost import XGBoost
from ..deep_learning.neuroevolution import NeuroEvolution
from ..deep_learning.particle_swarm_optimization import ParticleSwarmOptimizedNN
