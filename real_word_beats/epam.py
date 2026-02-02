"""  20260128 

Given a PySpark DataFrame test_df with columns: 
    product_code(string), base_date(datetime), and qty(int), use PySpark functions to implement the following requirements:
1. Extract the latest qty for each product_code;
You are expected to:
Use PySpark DataFrame and Window functions to perform all operations.
Ensure each of the operations is efficient and scalable for large datasets


Write a function to move all even digits in an integer to the end while preserving the order of digits.
def move_even_to_end(n: int) -> int:
    # Your implementation here
    pass
# Example:
# Input: 431265
# Output: 315426 (odd digits: 3, 1, 5; even digits: 4, 2, 6)

"""





def move_even_to_end(n:int)-> int:
    w = str(n)
    
    
    # for 
    
    pass 

s = 431265



# this is what i wrote....complete = =...
list_s = list(str(s))
result_even = [] 
result_odd = []
for index, string in enumerate(list_s): 
    if int(string) % 2 != 0:
       result_odd.append(int(string))
    if int(string) % 2 == 0:
       result_even.append(int(string))

# print(result_odd, result_even)
result = "".join(str(result_odd + result_even))
    

print(result)
        


# move_even_to_end(s)



