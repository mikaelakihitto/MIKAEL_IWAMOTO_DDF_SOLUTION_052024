import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

def predict(input):
    st.sidebar.header("Prediction")
    data = input
    #predict_column = st.sidebar.selectbox("Select value to predict", data.columns, index=data.columns.get_loc("SalePrice"))

    categorical = data.select_dtypes(include=['object'])
    non_categorical = data.select_dtypes(exclude=['object'])

    #Limpando dados 
    categorical = categorical.fillna('0')
    non_categorical = non_categorical.fillna(0)
    #aplicando one hot encoder
    categorical_encoder = pd.get_dummies(categorical)


    st.write(data.head())

    # Define the features (X) and target (y)
    # separando feature e target
    X = pd.concat([categorical_encoder , non_categorical],axis =1)
    X = X.drop('SalePrice',axis=1)
    y = non_categorical['SalePrice']

    # Split the data into train and test sets (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize the RandomForestRegressor model
    model = RandomForestRegressor(n_estimators=100, random_state=42)

    # Train the model using the training data
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    # Avaliando o desempenho do modelo r2score
    r2 = r2_score(y_test, y_pred)
    st.text("r2_score do modelo: {}".format(r2))

    ## Obtendo a importancia das features
    importances = model.feature_importances_
    feature_importance_map = dict(zip(X.columns,importances))
    feature_importance_map = sorted(feature_importance_map.items(),key=lambda x: x[1],reverse= True)
   
    #printando as features que representam 95% do modelo
    st.text("AS FEATURES QUE REPRESENTAM 95% DO MODELO (UTILIZANDO RANDOM FOREST)")
    a=0
    for feature in feature_importance_map:
        st.write(feature)
        a=a+feature[1]
        if a >0.95:
            break

    #mostrando a analise de dispersão
    #residual = pd.DataFrame(y_test - y_pred)

    st.write("Grafico de Dispersão: Resposta do Real x Resposta do Modelo")
    fig, ax = plt.subplots()
    sns.scatterplot(x=y_test, y=y_pred, ax=ax)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90, ha="right")
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True, nbins=10))
    st.pyplot(fig)

def eda(data):
    st.sidebar.header("Visualizations")

    st.header("Upload your CSV data file")
    data_file = "/Users/mh/Downloads/movieData/movies.csv"
    # st.file_uploader("Upload CSV", type=["csv"])
    if data is not None:
        # data = pd.DataFrame(records, columns=keys)
        st.write("Data overview:")
        st.write(data.head())

        plot_options = ["Bar plot", "Scatter plot", "Histogram", "Box plot"]
        selected_plot = st.sidebar.selectbox("Choose a plot type", plot_options)

        if selected_plot == "Bar plot":
            x_axis = st.sidebar.selectbox("Select x-axis", data.columns)
            y_axis = st.sidebar.selectbox("Select y-axis", data.columns)
            st.write("Bar plot:")
            fig, ax = plt.subplots()
            sns.barplot(x=data[x_axis], y=data[y_axis], ax=ax)
            ax.set_xticklabels(ax.get_xticklabels(), rotation=90, ha="right")
            ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True, nbins=10))
            st.pyplot(fig)

        elif selected_plot == "Scatter plot":
            x_axis = st.sidebar.selectbox("Select x-axis", data.columns)
            y_axis = st.sidebar.selectbox("Select y-axis", data.columns)
            st.write("Scatter plot:")
            fig, ax = plt.subplots()
            sns.scatterplot(x=data[x_axis], y=data[y_axis], ax=ax)
            ax.set_xticklabels(ax.get_xticklabels(), rotation=90, ha="right")
            ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True, nbins=10))
            st.pyplot(fig)

        elif selected_plot == "Histogram":
            column = st.sidebar.selectbox("Select a column", data.columns)
            bins = st.sidebar.slider("Number of bins", 5, 100, 20)
            st.write("Histogram:")
            fig, ax = plt.subplots()
            sns.histplot(data[column], bins=bins, ax=ax)
            ax.set_xticklabels(ax.get_xticklabels(), rotation=90, ha="right")
            ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True, nbins=10))
            st.pyplot(fig)

        elif selected_plot == "Box plot":
            column = st.sidebar.selectbox("Select a column", data.columns)
            st.write("Box plot:")
            fig, ax = plt.subplots()
            sns.boxplot(data[column], ax=ax)
            ax.set_xticklabels(ax.get_xticklabels(), rotation=90, ha="right")
            ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True, nbins=10))
            st.pyplot(fig)

pages = {"EDA":eda, "Predict":predict}

def main():
    st.title("Analise Imobiliaria! EDA Streamlit App")

    st.header("Upload your CSV data file")
    data_file = st.file_uploader("Upload CSV", type=["csv"])
    selected_page = st.sidebar.selectbox("Choose a page", options=list(pages.keys()))

    if data_file is not None:
        data = pd.read_csv(data_file)
    else:  # Add this block
				# Replace with your fixed file path
        fixed_file_path = "houseprice.csv"  
        data = pd.read_csv(fixed_file_path)

    pages[selected_page](data)

if __name__ == "__main__":
    main()