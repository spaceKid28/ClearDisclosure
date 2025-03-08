from src.RAGbot import RAGbot
import subprocess
def main():

    # this converts our libary of PDFs into a Vector Database and list of strings, 
    # both of which can be found in the output folder in this repo
    bot = RAGbot()
    bot.clean()
    # Run the hypercorn command
    subprocess.run(["hypercorn", "src.webapp:app", "--reload"])
    return
    # query = "What Strategic Risks were highlighted in Walmarts 2024 10-K report?"
    # print(answer_question(query))
if __name__ == '__main__':
    main()
