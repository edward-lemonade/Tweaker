import 'package:flutter/material.dart';
import 'screens/play.dart';

void main() {
	runApp(const MyApp());
}

class MyApp extends StatelessWidget {
	const MyApp({super.key});

	@override
	Widget build(BuildContext context) {
		return MaterialApp(
			title: 'Flutter Demo',
			theme: ThemeData(
				colorScheme: ColorScheme.fromSeed(seedColor: const Color.fromARGB(255, 38, 98, 15)),
			),
			home: const PlayPage(title: 'Flutter Demo Home Page'),
		);
	}
}