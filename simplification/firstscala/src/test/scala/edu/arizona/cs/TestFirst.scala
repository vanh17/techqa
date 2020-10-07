package edu.arizona.cs

import org.scalatest._

class TestFirst extends FlatSpec with Matchers {
	"basic math" should "support addition in Scala" in {
		val sum = 1 + 1
		sum should be (2)
	}

	it should "support multiplication too" in {
		val prod = 2 * 2
		prod should be (4)
	}
}