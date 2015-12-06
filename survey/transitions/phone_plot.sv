digraph finite_state_machine {
	rankdir=LR;
	size="8,5"
	node [shape = circle];
	start -> screen-pos [ label = "0.0611476952" ];
	start -> price-pos [ label = "0.074317968" ];
	start -> keyboard-pos [ label = "0.0809031044" ];
	start -> price-neg [ label = "0.1166509878" ];
	start -> call-neg [ label = "0.062088429" ];
	start -> servic-neg [ label = "0.0517403575" ];
	screen-pos -> app-pos [ label = "0.3993174061" ];
	screen-pos -> storage-pos [ label = "0.0546075085" ];
	app-pos -> app-pos [ label = "0.0676328502" ];
	app-pos -> price-pos [ label = "0.4589371981" ];
	app-pos -> battery-pos [ label = "0.0869565217" ];
	app-pos -> call-pos [ label = "0.0531400966" ];
	price-pos -> app-pos [ label = "0.0518518519" ];
	price-pos -> battery-pos [ label = "0.4222222222" ];
	battery-pos -> app-pos [ label = "0.052173913" ];
	battery-pos -> price-pos [ label = "0.0565217391" ];
	battery-pos -> sound-pos [ label = "0.4434782609" ];
	battery-pos -> call-pos [ label = "0.052173913" ];
	battery-pos -> cpu-pos [ label = "0.0695652174" ];
	sound-pos -> app-pos [ label = "0.0751445087" ];
	sound-pos -> camera-pos [ label = "0.3410404624" ];
	sound-pos -> servic-pos [ label = "0.0578034682" ];
	sound-pos -> cpu-pos [ label = "0.063583815" ];
	camera-pos -> app-pos [ label = "0.0764705882" ];
	camera-pos -> sound-pos [ label = "0.0705882353" ];
	camera-pos -> storage-pos [ label = "0.3705882353" ];
	camera-pos -> call-pos [ label = "0.0588235294" ];
	storage-pos -> sound-pos [ label = "0.0519480519" ];
	storage-pos -> call-pos [ label = "0.4242424242" ];
	storage-pos -> cpu-pos [ label = "0.0822510823" ];
	call-pos -> app-pos [ label = "0.067114094" ];
	call-pos -> battery-pos [ label = "0.0604026846" ];
	call-pos -> sound-pos [ label = "0.0738255034" ];
	call-pos -> servic-pos [ label = "0.3691275168" ];
	call-pos -> screen-neg [ label = "0.0536912752" ];
	servic-pos -> app-pos [ label = "0.0526315789" ];
	servic-pos -> cpu-pos [ label = "0.5448916409" ];
	cpu-pos -> app-pos [ label = "0.0787878788" ];
	cpu-pos -> price-pos [ label = "0.0727272727" ];
	cpu-pos -> battery-pos [ label = "0.0909090909" ];
	cpu-pos -> storage-pos [ label = "0.0545454545" ];
	cpu-pos -> keyboard-pos [ label = "0.3151515152" ];
	keyboard-pos -> battery-pos [ label = "0.0829694323" ];
	keyboard-pos -> call-pos [ label = "0.056768559" ];
	keyboard-pos -> design-pos [ label = "0.3886462882" ];
	design-pos -> app-pos [ label = "0.0543478261" ];
	design-pos -> camera-pos [ label = "0.0760869565" ];
	design-pos -> storage-pos [ label = "0.097826087" ];
	design-pos -> call-pos [ label = "0.0543478261" ];
	design-pos -> cpu-pos [ label = "0.0543478261" ];
	design-pos -> text-pos [ label = "0.3369565217" ];
	text-pos -> sound-pos [ label = "0.056" ];
	text-pos -> cpu-pos [ label = "0.08" ];
	text-pos -> screen-neg [ label = "0.384" ];
	screen-neg -> app-neg [ label = "0.3595505618" ];
	screen-neg -> battery-neg [ label = "0.0617977528" ];
	screen-neg -> sound-neg [ label = "0.0674157303" ];
	screen-neg -> camera-neg [ label = "0.0561797753" ];
	screen-neg -> storage-neg [ label = "0.0561797753" ];
	screen-neg -> servic-neg [ label = "0.0674157303" ];
	screen-neg -> cpu-neg [ label = "0.0561797753" ];
	screen-neg -> text-neg [ label = "0.0674157303" ];
	app-neg -> price-neg [ label = "0.5143884892" ];
	app-neg -> cpu-neg [ label = "0.0575539568" ];
	price-neg -> battery-neg [ label = "0.4729411765" ];
	price-neg -> sound-neg [ label = "0.0682352941" ];
	price-neg -> servic-neg [ label = "0.0635294118" ];
	price-neg -> cpu-neg [ label = "0.0658823529" ];
	battery-neg -> battery-neg [ label = "0.0940438871" ];
	battery-neg -> sound-neg [ label = "0.4858934169" ];
	battery-neg -> servic-neg [ label = "0.0626959248" ];
	battery-neg -> text-neg [ label = "0.0532915361" ];
	sound-neg -> battery-neg [ label = "0.0821256039" ];
	sound-neg -> camera-neg [ label = "0.38647343" ];
	sound-neg -> call-neg [ label = "0.0628019324" ];
	sound-neg -> servic-neg [ label = "0.0579710145" ];
	sound-neg -> text-neg [ label = "0.0676328502" ];
	camera-neg -> battery-neg [ label = "0.0640394089" ];
	camera-neg -> camera-neg [ label = "0.0689655172" ];
	camera-neg -> storage-neg [ label = "0.3201970443" ];
	camera-neg -> call-neg [ label = "0.0689655172" ];
	camera-neg -> cpu-neg [ label = "0.0541871921" ];
	camera-neg -> text-neg [ label = "0.0935960591" ];
	storage-neg -> price-neg [ label = "0.0506912442" ];
	storage-neg -> storage-neg [ label = "0.0552995392" ];
	storage-neg -> call-neg [ label = "0.4884792627" ];
	storage-neg -> text-neg [ label = "0.069124424" ];
	call-neg -> battery-neg [ label = "0.0826210826" ];
	call-neg -> servic-neg [ label = "0.566951567" ];
	call-neg -> cpu-neg [ label = "0.0883190883" ];
	servic-neg -> battery-neg [ label = "0.0638297872" ];
	servic-neg -> servic-neg [ label = "0.0744680851" ];
	servic-neg -> cpu-neg [ label = "0.5265957447" ];
	cpu-neg -> price-neg [ label = "0.0927152318" ];
	cpu-neg -> battery-neg [ label = "0.0728476821" ];
	cpu-neg -> storage-neg [ label = "0.0662251656" ];
	cpu-neg -> servic-neg [ label = "0.0529801325" ];
	cpu-neg -> cpu-neg [ label = "0.0529801325" ];
	cpu-neg -> keyboard-neg [ label = "0.3642384106" ];
	cpu-neg -> text-neg [ label = "0.059602649" ];
	keyboard-neg -> battery-neg [ label = "0.0670731707" ];
	keyboard-neg -> design-neg [ label = "0.4146341463" ];
	keyboard-neg -> text-neg [ label = "0.0609756098" ];
	design-neg -> price-neg [ label = "0.0732484076" ];
	design-neg -> battery-neg [ label = "0.0636942675" ];
	design-neg -> sound-neg [ label = "0.0605095541" ];
	design-neg -> text-neg [ label = "0.3885350318" ];
	text-neg -> price-neg [ label = "0.0777777778" ];
	text-neg -> battery-neg [ label = "0.0666666667" ];
	text-neg -> cpu-neg [ label = "0.0833333333" ];
	text-neg -> text-neg [ label = "0.0666666667" ];
}