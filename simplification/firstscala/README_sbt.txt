Hi all,

Here's a very quick set of instructions on how to use sbt for your project:

1. First, install sbt if you don't have it installed already. For this, follow the instructions in the Section 1.a of this document: 
https://www.scala-sbt.org/release/docs/Setup.html
Ignore everything else in the document! Also, I am assuming you have Java already installed. 
Note: If you chose to install sbt from one of the provided archives, add the bin/ directory in the sbt installation to your $PATH. 

Make sure your installation is Ok by typing:
	
	sbt sbtVersion

2. Download the skeleton project for your Scala project. I added it to D2L (under Content/Assignments). This is preconfigured to include the Lucene 7 dependencies. 
- Note that sbt requires that each project follows a very strict directory structure. This structure is identical to the one required by Maven. Let's explore it!
- Open the build.sbt file, to see how dependency configuration works in sbt. IMPORTANT NOTE: sbt will download the actual jar files for you. You simply tell it what you need in the build.sbt file.

3. OPTIONAL: The configuration in the skeleton project in D2L should be sufficient for HW3. If you want to add more library dependencies, first search for them on http://search.maven.org/, and then copy their information in the build.sbt file. Let's look at one example (processors).

4. Add your code to the cs.arizona.edu package. For example, I added the file src/main/scala/edu/arizona/cs/FirstScala.scala to get you started. 

5. Compile with the command:

	sbt compile 

Note that the first time you compile, it will take longer, because sbt will download all the dependencies you need. This happens just once. The result of the compilation is a target/ directory, which contains your classes.
Note: this is actually optional in sbt. sbt keeps track of change time stamps, and detects automatically when your code needs to be compiled, when you try to execute it.

6. Execute with the command:

	sbt 'runMain edu.arizona.cs.FirstScala example.txt'

Note that the sbt is more command line friendly than Maven. For example, you can include command line arguments in the runMain command, as shown above, without the need to create a profile.

You can adjust the command line arguments to the underlying java command in the file .sbtopts. Let's take a look.

7. OPTIONAL: other useful commands:
	sbt clean
	sbt run
	sbt test

