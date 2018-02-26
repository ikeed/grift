package com.grift.math.real;

import java.math.BigDecimal;
import java.math.BigInteger;
import java.math.MathContext;
import ch.obermuhlner.math.big.BigDecimalMath;
import ch.obermuhlner.math.big.BigFloat;
import org.apache.commons.math.util.MathUtils;
import org.jetbrains.annotations.NotNull;

public class Real {
    @NotNull
    private BigFloat.Context context = BigFloat.context(100);
    @NotNull
    private BigFloat value;
    @NotNull
    private BigFloat epsilon = context.valueOf(MathUtils.EPSILON);

    @NotNull
    public static final Real ONE = new Real(1);

    @NotNull
    public static final Real ZERO = new Real(0);

    public Real(char[] in, int offset, int len) {
        value = context.valueOf(new BigDecimal(in, offset, len));
    }

    public Real(char[] in, int offset, int len, MathContext mc) {
        value = context.valueOf(new BigDecimal(in, offset, len, mc));
    }

    public Real(char[] in) {
        value = context.valueOf(new BigDecimal(in));
    }

    public Real(char[] in, MathContext mc) {
        value = context.valueOf(new BigDecimal(in, mc));
    }

    public Real(String val) {
        value = context.valueOf(new BigDecimal(val));
    }

    public Real(String val, MathContext mc) {
        value = context.valueOf(new BigDecimal(val, mc));
    }

    public Real(double val) {
        value = context.valueOf(BigDecimal.valueOf(val));
    }

    public Real(double val, MathContext mc) {
        value = context.valueOf(new BigDecimal(val, mc));
    }

    public Real(BigInteger val) {
        value = context.valueOf(new BigDecimal(val));
    }

    public Real(BigInteger val, MathContext mc) {
        value = context.valueOf(new BigDecimal(val, mc));
    }

    public Real(BigInteger unscaledVal, int scale) {
        value = context.valueOf(new BigDecimal(unscaledVal, scale));
    }

    public Real(BigInteger unscaledVal, int scale, MathContext mc) {
        value = context.valueOf(new BigDecimal(unscaledVal, scale, mc));
    }

    public Real(int val) {
        value = context.valueOf(BigDecimal.valueOf(val));
    }

    public Real(int val, MathContext mc) {
        value = context.valueOf(new BigDecimal(val, mc));
    }

    public Real(long val) {
        value = context.valueOf(BigDecimal.valueOf(val));
    }

    public Real(long val, MathContext mc) {
        value = context.valueOf(new BigDecimal(val, mc));
    }

    public Real(BigDecimal value, BigFloat.Context context) {
        this.value = context.valueOf(value);
        this.context = context;
    }

    public Real(BigDecimal value) {
        this.value = context.valueOf(value);
    }

    public Real(Real copyMe) {
        this.context = copyMe.context;
        this.value = copyMe.value;
    }

    public static Real copyOf(Real copyMe) {
        return new Real(copyMe);
    }

    public void setDigitsPrecision(int digitsPrecision) {
        context = BigFloat.context(digitsPrecision);
        value = context.valueOf(value);
    }

    public int getDigitsPrecision() {
        return context.getPrecision();
    }

    public boolean isZero() {
        return BigFloat.abs(value).isLessThan(epsilon);
    }

    public Real ln() throws ArithmeticException {
        BigFloat x = context.valueOf(BigDecimalMath.log(value.toBigDecimal(), context.getMathContext()));
        return new Real(x.toBigDecimal(), context);
    }

    public Real subtract(Real v1) {
        return new Real(value.subtract(v1.value).toBigDecimal(), context);
    }

    public Real divide(Real v1) {
        return new Real(value.divide(v1.value).toBigDecimal(), context);
    }

    @Override
    public boolean equals(Object obj) {
        if (obj == null) {
            return false;
        } else if (obj instanceof Real) {
            return ((Real) obj).subtract(this).isZero();
        } else if (obj instanceof BigFloat) {
            return equals(new Real(((BigFloat) obj).toBigDecimal(), context));
        } else if (obj instanceof BigDecimal) {
            return equals(new Real((BigDecimal) obj, context));
        } else if (obj instanceof Double) {
            return equals(new Real((Double) obj, context.getMathContext()));
        }
        return false;
    }

    @Override
    public int hashCode() {
        return value.hashCode();
    }

    public Real add(Real w) {
        return new Real(context.valueOf(w.value.add(value)).toBigDecimal(), context);
    }

    public boolean isNegative() {
        return isLessThan(ZERO);
    }

    public boolean isLessThan(Real real) {
        return value.compareTo(real.value) < 0;
    }

    public boolean isGreaterThan(Real real) {
        return value.compareTo(real.value) > 0;
    }

    public Real multiply(Real real) {
        return new Real(context.valueOf(real.value.multiply(value)).toBigDecimal(), context);
    }

    @Deprecated
    public double multiply(double v) {
        return value.multiply(v).toDouble();
    }

    @Deprecated
    public static Real valueOf(double v) {
        return new Real(v, new MathContext(100));
    }
}
