10 REM Fibonacci sequence
20 LET A = 0
30 LET B = 1
40 PRINT "Fibonacci sequence:"
50 FOR I = 1 TO 10
60   PRINT A
70   LET C = A + B
80   LET A = B
90   LET B = C
100 NEXT I
110 END
