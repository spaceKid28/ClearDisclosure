from src.RAGbot import RAGbot
import sys
def main():

    query = " ".join(sys.argv[1:]) # command line input for query
    print(query)
    bot = RAGbot()
    # this converts our libary of PDFs into a Vector Database and list of strings, 
    # both of which can be found in the output folder in this repo 
    # RAGbot.clean() 
    response = bot.answer_question(query)
    print(f"Here is the LLM Output: {response}")
    return
#    query = "What Strategic Risks were highlighted in Walmarts 2024 10-K report?"
    # print(answer_question(query))
if __name__ == '__main__':
    main()
