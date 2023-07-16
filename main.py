def convert_to_binary(input_num):
    new_binary_num=""
    while input_num > 0:
         binary_digit=input_num % 2
         new_binary_num=str(binary_digit) + new_binary_num
         input_num=input_num // 2
    return new_binary_num
x = 129
new_num =convert_to_binary(x)
print(new_num)

