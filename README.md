Abstract

Stock market prediction is an interest for many investors. With new technologies and the rapid spread of information through social media, an event in one place would affect the other parts of the world easily, but also poses the insights for better stock market prediction (IMF, 2023), such as the sentiment analysis on Twitter having been effective. This project attempts to predict the stock market prices with Recurrent Neural Network (RNN) based on the stock prices data taken from Yahoo finance. The performance of this model successfully accomplished NUMBERS% gains. We solely depend on the market prices as data for building the stock market prediction model, but the future work will incorporate other measures such as sentiments, and also various ways to trade including futures, options, etc.

Introduction

Stock market prediction has always been a topic of interest for investors, financial analysts, and researchers. In recent years, with the growth of machine learning and deep learning, the use of neural networks for stock market prediction has gained significant attention. Neural networks are computational models that can be trained to recognize patterns and trends in data and can be used to make predictions based on past performance.

The year 2023 has been filled with major financial crises. The bankruptcy of Silicon Valley Bank brought considerable repercussions to the finance system, sentiments, and psychological behavior. IMF issued a Global Financial Stability Report (GFSR) in April 2023, witnessing the changes in sentiments that were observed on Twitter as the results of people’s behavior and reactions “Negative sentiment on Twitter surged, and stock prices tanked after Silicon Valley Bank announced its securities losses.“ (IMF, 2023, p. 47). These behaviors observed on Twitter have been an indicator of stock market prediction, suggesting the need and spread of algorithmic trading presence.

Algorithmic trading tries to accommodate as much information as possible, such as news articles, price movement patterns and trends, sentiments as mentioned above, etc. With the development of computational analysis and algorithms, there is a huge movement towards understanding the stock market quantitatively more and more.

Automating the Prediction of stock market fluctuations has been a difficult matter for a lot of companies trying to do so, mainly because of the number of factors that can affect stock changes and make predicting them perfectly nearly impossible as presented in the “Black Swan Theory” and/or “random walk theory.”
For example, Eli Lily’s stock dived down after a fake Twitter account impersonating them tweeted with the promise of free insulin. Such events create the difficulty in predicting daily stock changes to the full accuracy. 

Another thing to mention is the role of psychological factors and sentiments plays in the stock market. The psychological behaviors are not depicted quantitatively, making it difficult to accommodate these qualitative information into the quantitative prediction algorithm. There has been an attempt to do so, such as observed correlation between sentiments and stock market price movements, economic indicators, and company financial reports (Shen & Shafiq, 2020). Many other psychological factors face the difficulty to quantitate into the prediction, such as biases and irrational decision making. 

With these factors playing a huge role in the stock market, there has been the observation that theories established in behavioral finance and economics do not depict what it is, such as efficient market theory, which has been the discrepancies between theory and the markets (Sharma & Kumar, 2019). As such, there is more need than ever for behavioral aspects of finance (Hirshleifer, 2015), and becoming more critical to explore the realms of human behavior and the cognitive biases in decision-making (Costa, Carvalho, & Moreira, 2018), with some past case study qualitative, personality traits in the financial context (De Bortoli, Da Costa, Goulart, & Campara, 2019) as well.

Method

We use Python for the prediction of stock markets. The real stock market data taken from Yahoo finance has been utilized in this research. The model uses a simple LSTM (Long Short-Term Memory) neural network. LSTM is a type of Recurrent Neural Network (RNN) that can capture long-range dependencies in a sequence, making it suitable for time series data like stock prices. 

Our main goal is to achieve a working model while maximizing accuracy. We will measure the performance of this model as well by deciding on buy/sell actions based on the comparison between the predicted value and actual market data.

To build our LSTM-based stock market predictor, we utilize Python and the PyTorch library. We employ historical stock market data from sources such as Yahoo Finance to train our model. Our LSTM network takes into account the stock's historical price, volume, and technical indicators as input features. It then predicts the stock's future closing price for the next  7 days.

The dataset is preprocessed, normalized, and divided into training and testing sets. We train our model using the training set, and then we evaluate its performance using the testing set. To measure the model's accuracy, we use evaluation metrics such as  Mean Squared Error (MSE).

During the training phase, we can experiment with various hyperparameters, such as the number of layers in the LSTM network, the number of hidden units, and the dropout rate. Tuning these hyperparameters might help us achieve better prediction results. Additionally, we can explore other optimization techniques, such as learning rate schedules and weight initialization methods, to further improve the model's performance.




Results

In our initial implementation of the LSTM-based stock market prediction model, we observed that the accuracy was not as high as anticipated. The reasons for this could be the limited amount of input features, the choice of hyperparameters, and the inherent complexity of stock market data. However, the preliminary results still demonstrated the potential of LSTM neural networks in capturing the patterns and trends in stock market time series data.

During the evaluation phase, we noticed that the model's predictions were more accurate for some stocks compared to others. This could be attributed to the varying levels of volatility and predictability in different stocks, as well as the specific timeframes chosen for training and testing. Furthermore, we observed that the model performed better in predicting the overall trend of stock prices rather than the exact values.

Conlcusion

While the initial implementation of our LSTM-based stock market prediction model did not yield the desired level of accuracy, there is room for improvement. The challenges associated with stock market prediction, such as the complex interactions between various factors influencing stock prices, make this task inherently difficult. However, by refining our model, incorporating additional features, and exploring advanced techniques, we expect to see a significant increase in prediction accuracy in the upcoming weeks.

Despite the lower-than-expected initial accuracy, our model demonstrated the potential of LSTM neural networks in capturing intricate patterns and trends in time series data. With further improvements, our model could become a valuable tool for investors, financial analysts, and researchers in making more informed decisions.

Discussion

The results of our stock market prediction model serve as a stepping stone for further exploration and development of LSTM-based models for financial time series forecasting. Our experience with this model has highlighted several areas of potential improvement, which could lead to a significant enhancement in prediction accuracy.

Firstly, incorporating a broader range of features could improve the model's ability to capture the complex relationships between different factors affecting stock prices. By considering factors such as social media sentiment, economic indicators, and company financial reports, our model may become more robust and better equipped to predict stock prices.

Secondly, ensemble methods could be employed to further improve prediction accuracy. Combining LSTM networks with other machine learning techniques or using multiple LSTM models with different input features could yield more accurate predictions by leveraging the strengths of each approach.

Another crucial aspect to consider is the role of uncertainty in stock market predictions. Bayesian neural networks, as mentioned in the literature review, can provide not only reasonable predictions but also quantify the uncertainty of those predictions. Incorporating uncertainty measures into our model can be valuable for investors as it offers an additional dimension for decision-making.

Lastly, addressing the issue of overfitting and exploring advanced regularization techniques could contribute to better generalization of the model. This would result in more reliable predictions when applied to new, unseen data.

Ethics

When considering the ethics of our project, it is important to address several questions and concerns. The question of whether we should even be attempting to predict the stock market is relevant. However, understanding the potential growth of a company and analyzing the audience's perception can be useful, as the stock market reflects various opinions and information.

There will inevitably be imperfections in our attempts to predict the stock market due to its inherently chaotic nature, as described by the "random walk theory." To handle potential appeals or mistakes, our focus will be on obtaining the most accurate average prediction line possible.

Bias is a concern in any dataset, and although the stock market data itself may not be directly biased, the actions of humans buying and selling stocks introduce some level of bias. We will not be making any specific efforts to minimize this bias, as it is already incorporated into the market's movements.

We anticipate that different industries and companies may exhibit different patterns in their stock movements. To address this, our initial focus will be on a few companies within a single industry, with the potential for expansion later on.

Misinterpretations of results can be an issue, particularly since stock market data often contains noise. To prevent this, we will emphasize the randomness present in financial markets and discuss established theories that highlight this randomness.

Lastly, the issue of privacy and anonymity is crucial. Stock market prices do not reflect individuals' personal information and remain anonymous, as there is no way to determine who is responsible for specific trades using only quantitative stock market data. This ensures that our project does not infringe on individuals' privacy or anonymity.

Future work and Reflection

To enhance the performance of our LSTM-based stock market prediction model, future work should focus on the following areas:

1- Fine-tuning the model's hyperparameters: Experimenting with different network architectures, hidden units, dropout rates, and other hyperparameters can potentially improve the model's prediction accuracy. In addition, some near-future adjustments we plan to make include but are not limited to: Changing the time-slot for the training data collection, experimenting with low-high volatility companies, adding sell/buy action recommendations, and including various trading ways, such as futures, options, etc.
2- Incorporating additional features: Considering factors such as social media sentiment, economic indicators, and company financial reports can make the model more robust and better equipped to predict stock prices. In addition, adding graphic plots would make understanding the results more efficient.
3- Exploring ensemble methods: Combining LSTM networks with other machine learning techniques or employing multiple LSTM models with different input features can lead to more accurate predictions.
4- Investigating uncertainty measures: Incorporating Bayesian neural networks or other methods to quantify the uncertainty of predictions can provide valuable insights for investors.
Addressing overfitting and advanced regularization: By tackling the issue of overfitting and exploring advanced regularization techniques, our model's predictions could become more reliable when applied to new, unseen data.
5- Performance comparison with other models: As we make improvements to our model, it is essential to compare its performance with other existing models, such as traditional statistical models or alternative machine learning approaches. This will provide valuable insights into the model's relative strengths and weaknesses and guide further refinements.  
6- Exploring alternative deep learning architectures: In addition to refining the existing LSTM model, it would be beneficial to investigate other deep learning architectures, such as Transformer models or Convolutional Neural Networks (CNNs), to determine whether they can yield better results for stock market prediction.  
7- Personalized stock market predictions: Once our model's performance has been optimized, we could explore the potential of personalizing stock market predictions for individual investors. This would involve tailoring the model to consider specific investment goals, risk tolerance, and other individual factors, allowing investors to make more informed decisions based on their unique circumstances.

8- Integrating the model into a user-friendly platform: As the final step, we can work on integrating our refined stock market prediction model into a user-friendly platform or application. This would enable a broader audience to access and benefit from our model's predictions, making the stock market more approachable and understandable for both seasoned investors and novices alike.

By focusing on these areas for future work, we can continue to refine and improve our stock market prediction model, ultimately providing valuable insights and tools to investors, financial analysts, and researchers in the field of finance.

Looking back on the project, we recognize several areas where we could make improvements in future iterations. Firstly, we would invest more time and effort in exploring additional data sources such as social media sentiment, economic indicators, and company financial reports to build a more comprehensive dataset. This could potentially lead to a more accurate prediction model. Secondly, experimenting with different model architectures and techniques could have potentially improved the performance of our stock market predictor. In the future, we would consider exploring alternative deep learning architectures, such as transformers or attention mechanisms, and incorporating ensemble methods to enhance the model's performance. Finally, we would allocate more time for fine-tuning hyperparameters, as this could have a significant impact on the accuracy of our predictions.
 
Works Cited

Chandra, R., & He, Y. (2021). Bayesian neural networks for stock price forecasting before and during COVID-19 pandemic. PLOS ONE, 16(7). https://doi.org/10.1371/journal.pone.0253217 

Costa, D. F., Carvalho, F. de, & Moreira, B. C. (2018). Behavioral Economics and behavioral finance: A bibliometric analysis of the scientific fields. Journal of Economic Surveys, 33(1), 3–24. https://doi.org/10.1111/joes.12262 

Das, D., & Shorif Uddin, M. (2013). Data Mining and neural network techniques in stock market prediction : A methodological review. International Journal of Artificial Intelligence & Applications, 4(1), 117–127. https://doi.org/10.5121/ijaia.2013.4109 

De Bortoli, D., da Costa, N., Goulart, M., & Campara, J. (2019). Personality traits and investor profile analysis: A behavioral finance study. PLOS ONE, 14(3). https://doi.org/10.1371/journal.pone.0214062 

Devadoss, V. A., & Ligori, A. A. (2013). Stock prediction using Artificial Neural Networks. International Journal of Web Technology, 002(002), 45–51. https://doi.org/10.20894/ijwt.104.002.002.005 

Hirshleifer, D. (2015). Behavioral finance. Annual Review of Financial Economics, 7(1), 133–159. https://doi.org/10.1146/annurev-financial-092214-043752 

huseinzol05. (n.d.). Huseinzol05/stock-prediction-models: Gathers Machine Learning and deep learning models for stock forecasting including trading bots and simulations. GitHub. Retrieved April 18, 2023, from https://github.com/huseinzol05/Stock-Prediction-Models 

IMF. (2023, April 11). Global Financial Stability Report, April 2023. Global Financial Stability Report. Retrieved April 13, 2023, from https://www.imf.org/en/Publications/GFSR/Issues/2023/04/11/global-financial-stability-report-april-2023 

Naeini, M. P., Taremian, H., & Hashemi, H. B. (2010). Stock market value prediction using neural networks. 2010 International Conference on Computer Information Systems and Industrial Management Applications (CISIM). https://doi.org/10.1109/cisim.2010.5643675 

Originate. (n.d.). Retrieved February 28, 2023, from https://www.originate.com/thinking/predicting-stock-market-movements-using-a-neural-network https://github.com/Originate/dbg-pds-tensorflow-demo 

RK, D., & DD, P. (2010). Application of artificial neural network for stock market predictions: A review of literature. International Journal of Machine Intelligence, 2(2), 14–17. https://doi.org/10.9735/0975-2927.2.2.14-17 

Sharma, A., & Kumar, A. (2019). A Review Paper on Behavioral Finance: Study of Emerging Trends. Qualitative Research in Financial Markets, 12(2), 137–157. https://doi.org/10.1108/qrfm-06-2017-0050 

Shen, J., & Shafiq, M. O. (2020). Short-term stock market price trend prediction using a comprehensive deep learning system.Journal of Big Data, 7 (1). https://doi.org/10.1186/s40537-020-00333-6 

Vijh, M., Chandola, D., Tikkiwal, V. A., & Kumar, A. (2020). Stock closing price prediction using Machine Learning Techniques. Procedia Computer Science, 167, 599–606. https://doi.org/10.1016/j.procs.2020.03.326 

Vonko, D. (2022, July 8). Neural networks: Forecasting profits. Investopedia. Retrieved February 28, 2023, from https://www.investopedia.com/articles/trading/06/neuralnetworks.asp 

YouTube. (2022, May 23). Predict the stock market with machine learning and python. YouTube. Retrieved February 28, 2023, from https://www.youtube.com/watch?v=1O_BenficgE&ab_channel=Dataquest 

 



