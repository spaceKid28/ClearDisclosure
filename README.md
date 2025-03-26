README.md

Name: Bennett Lincoln
Project: RAG Agent SEC Filings

RUNNING THE PROJECT LOCALLY: 

This project must be run locally. After the repo has been cloned, using your package manager of choice, please install the required libraries listed in environment.yml. Below is some sample code that I would use, using Conda as a package manager:

    conda env create -f environment.yml
    conda activate clearDis

Note that I have named this environment clearDis in the requirements.yml file

Next, you must download the PDFs of the Financial disclosures from a google drive link here: https://drive.google.com/drive/folders/1C60vWf37T8fjHsqSl1E1bdcGd8YpL2xG?usp=drive_link

Put these PDF files into a folder named "data" in the ClearDisclosure repository, at the same level as the folders "src" and "templates"

Finally, run this command from the terminal to run the application: "python runner.py"

The website should run locally and you should be able to access it using any web broswer (I use chrome): http://127.0.0.1:8000/

Feel free to submit a query on the local webpage. The webpage should update in less than 90 seconds. After viewing the results, you can navigate back to the home page to submit another query by scrolling to the bottom and clicking the button.


README:

This is my implementation of a RAG agent that can query a database of financial documents. The idea is to reduce the manual labor of humans pouring through hundreds or thousands of pages of legalese. 

The user will submit a Query on the webpage, such as: "What are some of Walmarts biggest strategic risks highlighted in their 10-K in 2024" and the application will convert that query into an vector representation, perform similarity search in the vector database of available Financial disclosures, returning the 3 most relevant "chunks" (200 tokens) of text.

After the user has typed their query in the webpage and clicked submit, the webpage should reload and return the 3 most similar "chunks" to answer the question, along with the names of the Financial Documents the Chunks were pulled from. 

EXTRA FUNCTIONALITY:

I added a method to my RAGbot class that takes the relevant document chunks as context to LLM.  I used Microsoft's phi-2 model for mostly two reasons: a. the model is relatively small with only 2.7 Billion Parameters, and b. it is free and does not require any API tokens to access. Assuming the user creates their virtual environment from the environment.yml file, it should work correctly. 

I was not able to get the LLM to output correctly in the FASTAPI. However, it worked on my machine with the following command from the terminal. As long as the first two commands are python llm.py, you can replace the rest with your query.

    python llm.py What Strategic Risks were highlighted in Walmarts 2024 10-K report

Feel free to try and generate a LLM response, but the time it will take to run inference on the model will be dependent on your system. On my laptop it takes about 30 minutes to generate a reponse, but response times varied wildly during testing. 

Here are some of the sources:
(10-Ks and 10-Qs from publically traded companies) from https://www.bamsec.com/

Vector Database I used for similarity search came from Meta: 
https://engineering.fb.com/2017/03/29/data-infrastructure/faiss-a-library-for-efficient-similarity-search/

Some other links I used while developing my code for this project:

RAG implementation in Python (I ignored the SQL component in the below link):
https://aws.amazon.com/blogs/machine-learning/build-a-robust-text-to-sql-solution-generating-complex-queries-self-correcting-and-querying-diverse-data-sources/

Another project that implements RAG: 
https://github.com/aws-samples/natural-language-querying-of-data-in-s3-with-athena-and-generative-ai-text-to-sql

Another project that I used for reference:
https://github.com/Future-House/paper-qa?tab=readme-ov-file

Here is the terminal output from the LLM, it took 35 minutes to run on my Machine:

(clearDis) bennettlincoln@LAPTOP-PA16T0DR:~/repos/python_24/ClearDisclosure$ python llm.py What Strategic Risks were highlighted in Walmarts 2024 10-K report


CONTEXT: Results for query: 'What Strategic Risks were highlighted in Walmarts 2024 10-K report'

Result 1 (from 2024 10-K – Walmart Inc. – BamSEC.pdf):
A description of any substantive amendment or waiver of Walmart's Reporting Protocols for Senior Financial Officers or our Code of
Conduct for our chief executive officer, our chief financial officer and our controller, who is our principal accounting officer, will be
disclosed on our website at www.stock.walmart.com under the Corporate Governance section. Any such description will be located on
our website for a period of 12 months following the amendment or waiver.
ITEM 1A.RISK FACTORS
The risks described below could, in ways we may or may not be able to accurately predict, materially and adversely affect our business,
results of operations, financial position and liquidity. Our business operations could also be affected by additional factors that apply to
all companies operating in the U.S. and globally. The following risk factors do not identify all risks that we may face.
Strategic Risks

Result 2 (from Q2'25 10-Q – Walmart Inc. – BamSEC.pdf):
323/2/25, 4:06 PM wmt-20240731
https://www.bamsec.com/filing/10416924000141?cik=104169 40/44
Table of Contents
Risks, Factors and Uncertainties Regarding Our Business
These forward-looking statements are subject to risks, uncertainties and other factors, domestically and internationally, including:
Economic Factors
•economic, geopolitical, capital markets and business conditions, trends and events around the world and in the markets in
which Walmart operates;
•currency exchange rate fluctuations;
•changes in market rates of interest;
•inflation or deflation, generally and in certain product categories;
•transportation, energy and utility costs;
•commodity prices, including the prices of oil and natural gas;
•changes in market levels of wages;
•changes in the size of various markets, including eCommerce markets;
•unemployment levels;
•consumer confidence, disposable income, credit availability, spending levels, shopping patterns, debt levels, and demand for
certain merchandise;

Result 3 (from Q1'25 10-Q – Walmart Inc. – BamSEC.pdf):
condition or results of operations.
313/2/25, 4:06 PM wmt-20240430
https://www.bamsec.com/filing/10416924000105?cik=104169 38/42
Table of Contents
Risks, Factors and Uncertainties Regarding Our Business
These forward-looking statements are subject to risks, uncertainties and other factors, domestically and internationally, including:
Economic Factors
•economic, geopolitical, capital markets and business conditions, trends and events around the world and in the markets in
which Walmart operates;
•currency exchange rate fluctuations;
•changes in market rates of interest;
•inflation or deflation, generally and in certain product categories;
•transportation, energy and utility costs;
•commodity prices, including the prices of oil and natural gas;
•changes in market levels of wages;
•changes in the size of various markets, including eCommerce markets;
•unemployment levels;




Loading checkpoint shards: 100%|██████████████████████████████████████████████████████████████████████████████████| 2/2 [00:00<00:00,  2.75it/s]
Some parameters are on the meta device because they were offloaded to the disk and cpu.
Setting `pad_token_id` to `eos_token_id`:50256 for open-end generation.

The Strategic Risks highlighted in Walmarts 2024 10-K report are:
1. Economic Factors: The risks associated with economic factors include changes in market levels of wages, unemployment levels, consumer confidence, disposable income, credit availability, spending levels, debt levels, and demand for certain merchandise. These factors can affect the business operations of Walmart and may have a material and adverse impact on their financial position and liquidity.

2. Currency Exchange Rate Fluctuations: Walmart operates globally and is subject to currency exchange rate fluctuations. Any significant changes in exchange rates can affect the company's profitability and financial performance.

3. Changes in Market Rates of Interest: Changes in market rates of interest can impact Walmart's ability to borrow funds and affect their cost of capital. Higher interest rates can increase the cost of financing and may lead to lower profitability.

4. Inflation or Deflation, Generally and in Certain Product Categories: Inflation or deflation can impact the purchasing power of consumers
Program has completed
