# 20260124


from typing import List

def optiver_cnt(grid:List[List[str]]) -> int:
    target = "optiver"
    word_len = len(target)
    
    rows = len(grid)
    print(f'rows is : {rows}')
    cols = len(grid[0])
    print(f'col is {cols}')
    
    count = 0
    
    # count horizontally
    for r in range(rows):   # equal to "for r in range(0, rows)". if rows=2, then both produces 0,1
        for c in range(cols - word_len +1):
            if grid[r][c:c+word_len] == list(target):   #can also use all() here but not necessary. choose the simpliest way
                count += 1
    
    # count vertically
    for c in range(cols):
        for r in range(rows - word_len + 1):
            if all(grid[r + i][c] == target[i] for i in range(word_len)):   #all(): only return when all conditions are true
                count += 1
                
    return count



matrix = [
    ['o','p','t','i','v','e','r','f'],
    ['p','o','d','f','s','g','r','h'],
    ['t','g','s','d','u','y','t','j'],
    ['i','x','x','x','x','x','x','x'],
    ['v','x','x','x','x','x','x','x'],
    ['e','x','x','x','x','x','x','x'],
    ['r','x','x','x','x','x','x','x'],
    ['x','x','x','x','x','x','x','x'],
]
print
# print(optiver_cnt(matrix))