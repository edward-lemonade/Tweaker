import 'package:flutter/material.dart';
import 'screens/play.dart';
import 'services/hand_service.dart';

const cvDirectory = String.fromEnvironment('CV_SERVICE_DIR');
final handService = HandService(pythonDir: cvDirectory);

void main() {
  print('[main] CV_SERVICE_DIR: $cvDirectory');
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
			home: PlayScreen(handService: handService),
		);
	}
}