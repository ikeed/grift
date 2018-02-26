package com.grift.math;

import com.grift.forex.symbol.ImmutableSymbolIndexMap;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.springframework.test.context.junit4.SpringRunner;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotSame;

@RunWith(SpringRunner.class)
public class ProbabilityVectorFactoryTest {

    @Test
    public void testConstructWithMap() {
        ImmutableSymbolIndexMap map = new ImmutableSymbolIndexMap("CAD", "USD", "GBP");
        ProbabilityVector.Factory factory = new ProbabilityVector.Factory(map);
        assertEquals(3, factory.create().getDimension());
    }

    @Test
    public void testConstructWithStringVararg() {
        ProbabilityVector.Factory factory = new ProbabilityVector.Factory("CAD", "USD", "GBP");
        assertEquals(3, factory.create().getDimension());
    }

    @Test
    public void testCreateWithDoubleVararg() {
        ProbabilityVector.Factory factory = new ProbabilityVector.Factory("CAD", "USD", "GBP");
        ProbabilityVector probabilityVector = factory.create(0.2, 0.3, 0.5);
        assertEquals(3, probabilityVector.getDimension());
        assertEquals("[0.2, 0.3, 0.5]", probabilityVector.toString());
    }

    @Test(expected = IllegalStateException.class)
    public void testCopyWithWrongSymbols() {
        ProbabilityVector.Factory factory = new ProbabilityVector.Factory("CAD", "USD", "GBP");
        ProbabilityVector.Factory factory2 = new ProbabilityVector.Factory("CAD", "USD", "EUR");
        ProbabilityVector probabilityVector = factory.create(0.2, 0.3, 0.5);
        factory2.copy(probabilityVector);
    }

    @Test
    public void testCopyHappyPath() {
        ProbabilityVector.Factory factory = new ProbabilityVector.Factory("CAD", "USD", "GBP");
        ProbabilityVector probabilityVector = factory.create(0.2, 0.3, 0.5);
        ProbabilityVector result = factory.copy(probabilityVector);
        assertNotSame(probabilityVector, result);
        assertEquals("Vectors don't match", probabilityVector.toString(), result.toString());
    }


    @Test
    public void testNegativeElements() {
        ProbabilityVector.Factory factory = new ProbabilityVector.Factory("CAD", "USD", "GBP");
        ProbabilityVector probabilityVector = factory.create(-0.2, -0.3, -0.5);
        assertEquals("[0.2, 0.3, 0.5]", probabilityVector.toString());
    }
}