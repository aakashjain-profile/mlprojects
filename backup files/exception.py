from openai import OpenAI
import sys
from src.logger import logging  # Import the logger from logger.py

API_KEY = 'sk-345dee26c75243d480e6598b7a1de210'
API_URL = 'https://api.deepseek.com'

# Function to call OpenAI API
def call_openai_for_solution(exception_message):
    try:

        client = OpenAI(api_key=API_KEY, base_url=API_URL)

        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": "You are a helpful assistant"},
                {"role": "user", "content": "Hello"},
            ],
            stream=False
        )


        # # Using the OpenAI API to get a solution for the exception
        # response = openai.Completion.create(
        #     model="gpt-3.5-turbo",  # Using GPT-3.5 instead of GPT-4
        #     prompt=f"Here is an error message from Python: '{exception_message}'. Can you provide a solution or debugging advice for this exception?",
        #     max_tokens=150  # Adjust the token limit as needed
        # )
        openai_response = response.choices[0].message.content
        logging.error(f"OpenAI suggested solution: {openai_response}")
        return openai_response
    except Exception as e:
        logging.error(f"Error while calling OpenAI API: {e}")
        return None

# Error message details function
def error_message_detail(error, error_detail: sys):
    _, _, exc_tb = error_detail.exc_info()
    file_name = exc_tb.tb_frame.f_code.co_filename
    line_number = exc_tb.tb_lineno
    error_message = f"Error occurred in script [{file_name}] at line number [{line_number}] with error message: [{str(error)}]"
    return error_message

# CustomException class
class CustomException(Exception):
    def __init__(self, error_message, error_detail: sys):
        super().__init__(error_message)  # Initialize the base class with error_message
        self.error_message = error_message_detail(error_message, error_detail)

        # Log the exception details
        logging.error(f"Exception Type: {type(self)}")
        logging.error(f"Error Message: {self.error_message}")

        # Call OpenAI API to get a solution
        solution = call_openai_for_solution(self.error_message)

        if solution:
            logging.error(f"Suggested Solution from OpenAI: {solution}")

    def __str__(self):
        return self.error_message
