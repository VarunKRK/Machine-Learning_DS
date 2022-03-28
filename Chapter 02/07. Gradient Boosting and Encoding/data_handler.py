import pandas as pd

from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

hello = "<3"

def get_data(pth):

    data = pd.read_csv(pth)

    x_train, x_test, y_train, y_test = train_test_split(data.values[:,:-1], data.values[:,-1], test_size=0.2, random_state = 0)

    sc = StandardScaler()
    oe = OrdinalEncoder()

    cs = [0,2,3]
    ce = [1,4,5]

    p = Pipeline(
        [("coltransformer", ColumnTransformer(
            transformers=[
                ("varied", Pipeline([("scale", sc)]), cs),
                ("category", Pipeline([("encode", oe)]), ce),
            ]),
        )]
    )

    x_train = p.fit_transform(x_train)
    x_test = p.transform(x_test)


    #ct = ColumnTransformer( [('ordinal', OrdinalEncoder(handle_unknown= 'use_encoded_value', unknown_value = -1), [1,4,5] )] )
    
    # ct = ColumnTransformer( [('ordinal', OrdinalEncoder(handle_unknown= 'use_encoded_value', unknown_value = -1), [1,4,5] ),('non_transformed','passthrough',[0,2,3])] )


    # x_train = ct.fit_transform(x_train)
    # x_test = ct.transform(x_test)

    # sclr = StandardScaler()

    # x_train = sclr.fit_transform(x_train)
    # x_test = sclr.fit(x_train)



    #scale the x_train, x_test

    

    return x_train, x_test, y_train, y_test

    

x_train, x_test, y_train, y_test = get_data('insurance.csv')

print(x_test)