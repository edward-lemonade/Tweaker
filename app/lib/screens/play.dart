import 'package:flutter/material.dart';

class PlayPage extends StatefulWidget {
	const PlayPage({super.key, required this.title});

	final String title;

	@override
	State<PlayPage> createState() => _PlayPageState();
}

class _PlayPageState extends State<PlayPage> {
	int _counter = 0;

	void _incrementCounter() {
		setState(() {
			_counter++;
			_counter += 3;
		});
	}

	@override
	Widget build(BuildContext context) {
		return Scaffold(
			appBar: AppBar(
				backgroundColor: Theme.of(context).colorScheme.inversePrimary,
				title: Text(widget.title),
			),
			body: Center(
				child: Column(
					mainAxisAlignment: MainAxisAlignment.center,
					children: [
						const Text('You have pushed the button this many times:'),
						Text(
							'$_counter',
							style: Theme.of(context).textTheme.headlineMedium,
						),
					],
				),
			),
			floatingActionButton: FloatingActionButton(
				onPressed: _incrementCounter,
				tooltip: 'Increment',
				child: const Icon(Icons.add),
			),
		);
	}
}