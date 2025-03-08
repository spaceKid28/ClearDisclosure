from src.RAGbot import RAGbot
import sys
import os
def main():
    query = " ".join(sys.argv[1:]) # command line input for query

    # this converts our libary of PDFs into a Vector Database and list of strings, 
    # both of which can be found in the output folder in this repo
    bot = RAGbot()
    if not os.path.exists("./output/pdf_embeddings.index"): # if we haven't created the embeddings yet, we need to do that
        bot.clean()
    # sample Query: What Strategic Risks were highlighted in Walmarts 2024 10-K report
    print(bot.answer_question(query))
    print(f"Program has completed")
    return

if __name__ == '__main__':
    main()