package com.grift.math.real;


import java.math.BigDecimal;
import ch.obermuhlner.math.big.BigFloat;
import org.junit.Before;
import org.junit.Test;

import static com.grift.math.real.Real.ONE;
import static com.grift.math.real.Real.ZERO;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNotEquals;
import static org.junit.Assert.assertNotSame;
import static org.junit.Assert.assertTrue;

@SuppressWarnings({"EqualsBetweenInconvertibleTypes", "ObjectEqualsNull"})
public class RealTest {

    private Real real;

    @Before
    public void setup() {
        real = ZERO;
    }

    @Test
    public void testEqualsNull() {
        assertFalse(real.equals(null));
    }

    @Test
    public void testEqualsRealFalse() {
        assertFalse(real.equals(ONE));
    }
    
    @Test
    public void testEqualsRealTrue() {
        assertTrue(real.equals(ZERO));
    }

    @Test
    public void testEqualsBigFloatFalse() {
        assertFalse(real.equals(BigFloat.context(10).valueOf(1)));
    }

    @Test
    public void testEqualsBigFloatTrue() {
        assertTrue(real.equals(BigFloat.context(10).valueOf(0)));
    }

    
    @Test
    public void testEqualsBigDecimalFalse() {
        assertFalse(real.equals(new BigDecimal(1)));
    }

    @Test
    public void testEqualsBigDecimalTrue() {
        assertTrue(real.equals(new BigDecimal(0)));
    }

    @Test
    public void testEqualsDoubleFalse() {
        assertFalse(real.equals(1d));
    }

    @Test
    public void testEqualsDoubleTrue() {
        assertTrue(real.equals(0d));
    }

    @Test
    public void testEqualsWTFFalse() {
        assertFalse(real.equals("WTF!"));
    }

    @Test
    public void testHashCodeMatch() {
        assertEquals(new Real(10.12).hashCode(), new Real(10.12).hashCode());
    }

    @Test
    public void testHashCodeMismatch() {
        assertNotEquals(new Real(10.120000000000001).hashCode(), new Real(10.12).hashCode());
    }

    @Test
    public void testPrecisionSetGet() {
        Real real = new Real(10.2);
        real.setDigitsPrecision(21);
        assertEquals(21, real.getDigitsPrecision());
    }

    @Test
    public void testInverse() {
        assertEquals(Real.valueOf(3d/5d), Real.valueOf(5d/3d).inverse());
    }

    @Test
    public void testOpposite() {
        assertEquals(Real.valueOf(-4d), Real.valueOf(4d).opposite());
    }

    @Test
    public void testCopy() {
        Real input = Real.valueOf(10.12d);
        Real output = (Real) input.copy();
        assertNotSame(input, output);
        assertEquals(input, output);
    }
}