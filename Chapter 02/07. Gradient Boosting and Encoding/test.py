import data_handler as dh
import train as tm
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error

x_train, x_test, y_train, y_test = dh.get_data(r"D:\Machine-Learning_DS\Chapter 02\07. Gradient Boosting and Encoding\insurance.csv")
dt, rf, gbt = tm.train_models(x_train, y_train)

def pred(x_test, y_test):

    y_pred_dt = dt.predict(x_test)

    y_pred_rf = rf.predict(x_test)

    y_pred_gbt = gbt.predict(x_test)


    score_dt = mean_squared_error(y_pred_dt, y_test)

    score_rf = mean_squared_error(y_pred_rf, y_test)

    score_gbt = mean_squared_error(y_pred_gbt, y_test)



    return score_dt, score_rf, score_gbt

score_dt, score_rf, score_gbt = pred(x_test, y_test)
print(score_gbt)


