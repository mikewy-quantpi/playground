# Original dictionary
# original_dict = {'a': 1, 'b': 2, 'c': 3}

# Assigning part of the original dictionary to a new dictionary
# new_dict = original_dict.copy()
# new_dict = original_dict

# Modifying the new dictionary
# new_dict['a'] = 100

original_dict = {
    'a': [
        [1, 2]
    ]
}

new_dict = {}
new_dict['b'] = original_dict['a']
new_dict['b'] = [
    ['one', 'two']
]

# Printing both dictionaries
print("Original dictionary:", original_dict)
print("New dictionary:", new_dict)

