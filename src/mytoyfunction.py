from typing import AnyStr

def toyfunction(search_term:AnyStr) -> dict:
    '''
    get a string, serch in sparql query, output a dict with all results according to JSON:API formalism'''

    mydict = {
        "jsonapi": {"version": "1.1"},
        "data": [{'record_1':
                  {}
                  },
                  {'record_2':{}
                  }
        ]
    }


    return mydict