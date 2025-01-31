
import os
import secrets
from fastapi import FastAPI, Depends, status, HTTPException, Request
from fastapi.responses import JSONResponse
from typing import Annotated,AnyStr

from src import mytoyfunction

JSON_API_MEDIA_TYPE = "application/vnd.api+json"

app = FastAPI()

@app.get("/v1/sparql_search", status_code=status.HTTP_200_OK,response_class=JSONResponse)

async def sparql_search(term: AnyStr) -> JSONResponse:

    '''
    it takes as an input a request made by the user with a given term and provides as an outut an api response

    term: the term to search in the sparql query
    '''

    my_output = mytoyfunction.toyfunction(search_term=term)

    my_headers = {
        "Content-Type": JSON_API_MEDIA_TYPE,
    }


    return JSONResponse(content=my_output,headers=my_headers,media_type=JSON_API_MEDIA_TYPE)