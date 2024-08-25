def findEmpty(board):
    for i in range(len(board)):
        for j in range(len(board[0])):
            if(board[i][j]==0):
                return (i,j)
    return None

def isValid(board, num, pos):
    for i in range(len(board[0])):
        if(board[pos[0]][i]==num and i!=pos[1]):
            return False
        
    for i in range(len(board)):
        if(board[i][pos[1]]==num and i!=pos[0]):
            return False
      

    box_x = pos[1]//3
    box_y = pos[0]//3

    for i in range(box_y*3, box_y*3 + 3):
        for j in range(box_x * 3, box_x*3 + 3):
            if board[i][j] == num and (i,j) != pos:
                return False

    return True


def solve(board):
    find = findEmpty(board)
    if not find:
        return True
    else:
        x,y = find
    for i in range(1,10):
        if(isValid(board,i,(x,y))): 
            board[x][y] = i
            if(solve(board)):
                return True
            board[x][y] = 0
    return False

def getBoard(bo):
    solve(bo)
    return bo