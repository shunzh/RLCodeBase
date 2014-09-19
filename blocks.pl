move_to(A, B) :-
	A \== table,
	A \== B,
	on(A,X),
	clear(A),
	clear(B),
	retract(on(A,X)),
	assert(on(A,B)).

clear(floor).
clear(B) :- 
	not(on(_X,B)).
