fruit_dictionary = {"Carrot": "C",
                    "Tomato": "T",
                    "Grape": "G",
                    "Berries": "B",
                    "Peas": "P",
                    "Corn": "Cn",
                    }

def field_var(fruit,loc):
    return fruit_dictionary[fruit]+str(loc)

def get_name(abbr):
    for key in fruit_dictionary:
        if fruit_dictionary[key] == abbr:
            return key