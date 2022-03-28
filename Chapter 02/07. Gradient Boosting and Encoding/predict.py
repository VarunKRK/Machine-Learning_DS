import data_handler as dh
import train as tm

x_train, x_test, y_train, y_test = dh.get_data(r"D:\Machine-Learning_DS\Chapter 02\07. Gradient Boosting and Encoding\insurance.csv")
gbt = tm.train_models(x_train, y_train)


model = "Gradient Boosting Regressor"
while True:

    age = int(input("How old are you? \n"))
    child = int(input("How many children do you have? \n"))
    smoke = bool(input("Do you smoke? \n"))
    bmi = float(input("What's is your bmi? \n"))
    sex = str(input("male or female? \n"))
    region = str(input('Which region are you from? \n'))

    




    '''
    Preprocess
    predict
    
    '''
    
    #print("You are too fucked up 1 milly")
    

