DEF run m( 
    WHILE c( noMarkersPresent c) w( 
        IF c( rightIsClear c) i( 
            turnRight move 
        ) 
        IF c( rightIsBlocked c) i( 
            IF c( frontIsClear c) i( 
                move 
            ) 
            IF c( frontIsBlocked c) i( 
                turnLeft 
            ) 
        ) 
    ) 
    pickMarker 
    WHILE c( noMarkersPresent c) w( 
        IF c( rightIsClear c) i( 
            turnRight move 
        ) 
        IF c( rightIsBlocked c) i( 
            IF c( frontIsClear c) i( 
                move 
            ) 
            IF c( frontIsBlocked c) i( 
                turnLeft 
            ) 
        ) 
    ) 
    putMarker 
m)