from fastapi import FastAPI, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from starlette.requests import Request
from src.RAGbot import RAGbot

app = FastAPI()

templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
def welcome(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/submit", response_class=HTMLResponse)
def handle_form(request: Request, user_input: str = Form(...)):
    bot = RAGbot()
   
    response = bot.retrieve_documents(user_input)
    
    return templates.TemplateResponse("index.html", {"request": request, "response": response})