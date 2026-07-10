def countVowels(str):
    countV=0
    for char in str:
        if char.lower() in 'aeiou':
            countV+=1
    return countV
        
n_input = input("Enter String:")
strLength=len(n_input)
result = countVowels(n_input)
print(f"number of vowels in {n_input} = {result}")
print(f"number of consonants in {n_input} = {strLength - result}")
