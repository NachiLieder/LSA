import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import pandas as pd
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from sklearn.model_selection import train_test_split, KFold
#plot density function of df_best
import matplotlib.pyplot as plt
import seaborn as sns

# Set a theme for seaborn
sns.set_theme()

def rbf_kernel(x, gamma=0.1):
    return np.exp(-gamma * np.abs(x - np.mean(x))**2)
def exp_decay(x, tau=1):
    return np.exp(-x/tau)

def inverse_decay(X,tau):
    return 1 / (1 + X / tau)

def power_law_decay(X, tau=1):
    return (1 / (X + 1)) ** tau

def gaussian_decay(X, tau=1):
    return np.exp(-X**2 / (2 * tau**2))


def fit_selected_model(best,X,Y):
    X_transformed = np.empty_like(X)
    for i in range(X.shape[1]):

        X_transformed[:, i] = inverse_decay(X[:, i], tau=list(best.values())[i])
    model = sm.OLS(Y, X_transformed)
    results = model.fit()

    rmse = np.sqrt(np.mean((Y - results.predict(pd.DataFrame(X_transformed))) ** 2))
    return results, model, rmse
def objective(tau,X, Y, function_name):
    # X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=4)

    # transform each feature using its own RBF kernel
    X_transformed_train = np.zeros_like(X)
    X_transformed_test = np.zeros_like(X)
    for i in range(X.shape[1]):
        if function_name == 'exponential_decay':
            X_transformed_train[:, i] = exp_decay(X[:, i], tau=tau[i])
            X_transformed_test[:, i] = exp_decay(X[:, i], tau=tau[i])
        if function_name == 'inverse_decay':
            X_transformed_train[:, i] = inverse_decay(X[:, i], tau=tau[i])
            X_transformed_test[:, i] = inverse_decay(X[:, i], tau=tau[i])
        if function_name == 'power_law_decay':
            X_transformed_train[:, i] = power_law_decay(X[:, i], tau=tau[i])
            X_transformed_test[:, i] = power_law_decay(X[:, i], tau=tau[i])
        if function_name == 'gaussian_decay':
            X_transformed_train[:, i] = gaussian_decay(X[:, i], tau=tau[i])
            X_transformed_test[:, i] = gaussian_decay(X[:, i], tau=tau[i])

        # X_transformed_train[:, i] = 1 / (1 + X[:, i] / tau[i])
        # X_transformed_test[:, i] = 1 / (1 + X[:, i] / tau[i])

    X_transformed_train = sm.add_constant(X_transformed_train)
    X_transformed_test = sm.add_constant(X_transformed_test)

    ###### Linear regression
    # Add a column of ones to include an intercept in the model

    # fit ordinary least squares regression on transformed features
    model = sm.OLS(Y, pd.DataFrame(X_transformed_train))
    results = model.fit()
    ###################



    # Calculate root mean square error
    rmse = np.sqrt(np.mean((Y - results.predict(pd.DataFrame(X_transformed_test)))**2))
    return {'loss': rmse, 'status': STATUS_OK }


##############
#### MAIN ####
##############


max_evals = 200
folds = 5



df = pd.read_csv("data/post_processed/row_game_lineups_mmm.csv", index_col=0)
y = df['y']
del df['y']

X = df.values
Y = np.array(y)

# split to 5 kfolds and run the whole process on each fold:
list_of_results = []

kf = KFold(n_splits=folds, random_state=42, shuffle=True)
for train_index, test_index in kf.split(X):

    X_train, X_test = X[train_index], X[test_index]
    Y_train, Y_test = Y[train_index], Y[test_index]

    model = sm.OLS(Y_train, pd.DataFrame(X_train))
    results = model.fit()
    # Calculate root mean square error
    rmse_old = np.sqrt(np.mean((Y_train - results.predict(pd.DataFrame(X_train))) ** 2))
    # print(results.summary())
    print(rmse_old)



    # Define the hyperparameters' search space
    # sspace for exponential_decay
    space = [hp.uniform('tau'+str(i), 1, 30) for i in range(X.shape[1])]
    # Run the algorithm
    trials = Trials()
    exp_best = fmin(lambda tau: objective(tau, X_train,Y_train, 'exponential_decay'),
                space=space, algo=tpe.suggest, max_evals=max_evals, trials=trials)
    exp_best_loss = min(trial['result']['loss'] for trial in trials.trials)

    # sspace for inverse_decay
    space = [hp.uniform('tau'+str(i), 1, 30) for i in range(X.shape[1])]
    # Run the algorithm
    trials = Trials()
    inv_best = fmin(lambda tau: objective(tau, X_train,Y_train, 'inverse_decay'),
                space=space, algo=tpe.suggest, max_evals=max_evals, trials=trials)
    inv_best_loss = min(trial['result']['loss'] for trial in trials.trials)

    # sspace for power_law_decay
    space = [hp.uniform('tau'+str(i), 0.1, 5) for i in range(X.shape[1])]
    # Run the algorithm
    trials = Trials()
    power_best = fmin(lambda tau: objective(tau, X_train,Y_train, 'power_law_decay'),
                space=space, algo=tpe.suggest, max_evals=max_evals, trials=trials)
    power_best_loss = min(trial['result']['loss'] for trial in trials.trials)

    # sspace for gaussian_decay
    space = [hp.uniform('tau'+str(i), 10, 100) for i in range(X.shape[1])]
    # Run the algorithm
    trials = Trials()
    gauss_best = fmin(lambda tau: objective(tau, X_train,Y_train, 'gaussian_decay'),
                space=space, algo=tpe.suggest, max_evals=max_evals, trials=trials)
    gauss_best_loss = min(trial['result']['loss'] for trial in trials.trials)



    list_of_results.append([ exp_best_loss,inv_best_loss,power_best_loss,gauss_best_loss,rmse_old])
    # # print(best)
    # # plot densirty of the trials
    # plt.hist([t['result']['loss'] for t in trials.trials], bins=20)
    # plt.show()
    #
    # #convert dictionary to dataframe
    # df_best = pd.DataFrame(best, index=[0]).T
    # # Create a density plot using seaborn for a nicer output
    # for col in df_best.columns:
    #     sns.kdeplot(df_best[col], fill=True, legend=True)
    #
    # # Add a legend and title
    # plt.legend(title=r'$\tau_i$', title_fontsize='13', fontsize='12')
    # plt.title('Density plot of best hyperparameters', size=15)
    #
    # # Show the plot
    # plt.show()



    # results , model  , rmse_imp = fit_selected_model(best, X, Y)
    # print(results.summary())
    # print(rmse_imp)
    print('---------------')


df_results = pd.DataFrame(list_of_results, columns = ['exponential_decay','inverse_decay','power_law_decay','gaussian_decay','baseline'])
df_results.to_csv('decay_look_over_elastic.csv')
print()
