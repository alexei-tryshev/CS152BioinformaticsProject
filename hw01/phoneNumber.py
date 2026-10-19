import re

def isPhoneNumber(pattern, number):
    if re.search(pattern, number):
        return True
    else:
        return False

pattern = r"^\d{3}-\d{3}-\d{4}"

number1 = "408-876-7529" #true
number2 = "123-456-7890" #true
number3 = "123-3456-019" #false

print("number1 is a phone number: ", isPhoneNumber(pattern, number1))
print("number2 is a phone number: ", isPhoneNumber(pattern, number2))
print("number3 is a phone number: ", isPhoneNumber(pattern, number3))