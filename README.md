# ClearDisclosure
Name: Bennett Lincoln
Proposal: RAG Agent SEC Filings

I want to create a RAG agent that can query a database of financial documents. The idea is to make it easier for journalists to hold corporations accountable, and reduce the manual labor of humans pouring through hundreds or thousands of pages of legalese. 

The user will submit a prompt to the Agent; the answer should exist within the database.

Sample Prompts: How much did Walmart pay in corporate taxes in 2022? Did Williams Sonoma ever disclose how much of their cost of goods sold is imported from China? How much did ExxonMobile’s PAC contribute to political campaigns in 2018?

I would start by downloading Yearly and Quarterly Disclosures (10-k and 10-Q) from 5 or 6 publically traded companies for the past 5 years (I would start with Walmart, Exxon Mobile, Amazon, ect…) and store them in a document database. FIASS, which is provided by Meta, seems like a good database option. 

https://engineering.fb.com/2017/03/29/data-infrastructure/faiss-a-library-for-efficient-similarity-search/

Some other links that outline the more complex parts of the project:

RAG implementation in Python (I will ignore the SQL component in the below link):
https://aws.amazon.com/blogs/machine-learning/build-a-robust-text-to-sql-solution-generating-complex-queries-self-correcting-and-querying-diverse-data-sources/

Another project that implements RAG: 
https://github.com/aws-samples/natural-language-querying-of-data-in-s3-with-athena-and-generative-ai-text-to-sql

Finally, where I will pull the publicly available financial documents: https://www.bamsec.com/


Notes from Professor Troy:
I recommend taking a look at recent research on Supervised Fine Tuning (SFT). The following paper compares the two approaches on a specific problem focused on generating magnesium alloys, but the conclusions might generalize (and in fact you could evaluate that in this project, potentially) https://onlinelibrary.wiley.com/doi/10.1002/mgea.77

Here's an audio clip of Malcolm Diggs explaining that paper, in case you have limited patience for reading academic language: https://s3.amazonaws.com/journalclub.io/pdgpt-sft-full.mp3


Project Plan
By the end of week 4, I will have explored web frameworks and made a basic web app that says “hello” 
By the end of week 5, I will have a basic homepage, and I will have a data model that allows me to add cupcakes, which appear in a list on the homepage.
By the end of week 6, I will have it so you can click on a cupcake, enter my name, enter a rating for that cupcake, and enter notes about that cupcake.
By the end of week 7, I will add an “ingredients” section to my cupcakes, and I will be researching how I want to aggregate the ingredients data
By the end of week 8, I will have the “my preferences” page which ranks which ingredients are my favorites based on how I am rating the cupcakes that contain them. 
Finally, I will add some messaging to help people use my site, like “Sorry, you have to rate at least 5 cupcakes before we can calculate your preferences”

Project Plan
Week 1: Document Collection & Basic Processing

Set up development environment
Manually Download 10-K and 10-Q documents from ONE company (start with Walmart), from last year (10-Ks and 10-Qs) from https://www.bamsec.com/
Create simple script to convert PDFs to text
Basic text cleaning (remove headers, page numbers)
End goal: Have clean text files from one company's financial documents stored in github repo

Week 2: Document Database Setup

Research and understand FAISS setup requirements
Create document database structure
Implement basic document storage
Create simple script to add "cleaned" text documents to FAISS database
End goal: Financial Documents for Walmart stored in FAISS with basic metadata

Week 3: Text Chunking & Embeddings

Implement document chunking strategy
Test different chunk sizes
Create embeddings pipeline
Store embedded chunks in FAISS
End goal: Ability to create and store embeddings for document chunks

Week 4: Basic Retrieval (This will probably be the hardest for me)

Implement similarity search
Create simple script to query the database
Test retrieval quality with sample questions
Tune chunk size and similarity thresholds
End goal: Can retrieve relevant document chunks for test questions

Week 5: LLM Integration

Set up LLM API connection (ChatGPT or Anthropic, probably Anthropic API)
Create basic prompt template
Implement simple RAG pipeline
Test with sample financial questions
End goal: System can answer basic questions about financial data using RAG

