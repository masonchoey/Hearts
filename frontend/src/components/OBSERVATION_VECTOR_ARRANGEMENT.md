1. Pass Direction (4 values)
One-hot encoding indicating the passing direction
Values: No Pass (0), Left (1), Across (2), Right (3)

2. Dealt Hand (52 values)
Binary encoding of the cards initially dealt to the player
Each card position is 1 if dealt to this player, 0 otherwise

3. Passed Cards (52 values)
Binary encoding of the 3 cards the player passed away
Each card position is 1 if the player passed that card, 0 otherwise

4. Received Cards (52 values)
Binary encoding of the 3 cards received from another player
Only populated once all players have completed passing
Each card position is 1 if received, 0 otherwise

5. Current Hand (52 values)
Binary encoding of the cards currently in the player's hand
Each card position is 1 if currently held, 0 otherwise

6. Point Totals (144 values = 36 × 4 players)
Thermometer representation of each player's score
Each player gets 36 values (to account for range from -10 to +26)
For a score of X points, the first (X + 10) values are 1, rest are 0
This accounts for the Jack of Diamonds (-10 points) penalty

7. History of Tricks (4,732 values = 13 tricks × 364)
Each trick uses 364 values = 52 cards × 7 positions
The 7 positions represent: N E S W N E S (circular play pattern)
For each position, a one-hot encoding indicates which card was played
The layout accounts for which player led (starting position shifts accordingly)
