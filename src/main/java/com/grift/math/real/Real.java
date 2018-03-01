package com.grift.math.real;

import java.math.BigDecimal;
import java.math.MathContext;
import ch.obermuhlner.math.big.BigDecimalMath;
import ch.obermuhlner.math.big.BigFloat;
import org.apache.commons.math.util.MathUtils;
import org.jetbrains.annotations.NotNull;
import org.jscience.mathematics.structure.Field;

import static lombok.Lombok.checkNotNull;

@SuppressWarnings("WeakerAccess")
public class Real implements Field<Real> {
    @NotNull
    private BigFloat.Context context = BigFloat.context(100);
    @NotNull
    private BigFloat value;
    @NotNull
    private static final BigFloat EPSILON = BigFloat.context(100).valueOf(MathUtils.EPSILON);

    @NotNull
    public static final Real ONE = new Real(1);

    @NotNull
    public static final Real ZERO = new Real(0);

    public Real(double val) {
        value = context.valueOf(BigDecimal.valueOf(val));
    }

    public Real(double val, @NotNull MathContext mc) {
        value = context.valueOf(BigDecimal.valueOf(val));
        context = BigFloat.context(mc);
    }

    public Real(int val) {
        value = context.valueOf(BigDecimal.valueOf(val));
    }

    public Real(@NotNull BigDecimal value, @NotNull BigFloat.Context context) {
        this.value = context.valueOf(value);
        this.context = context;
    }

    public Real(@NotNull Real copyMe) {
        this.context = copyMe.context;
        this.value = copyMe.value;
    }

    public Real(@NotNull BigFloat value, @NotNull MathContext mathContext) {
        this.value = checkNotNull(value, "value");
        this.context = BigFloat.context(checkNotNull(mathContext, "mathContext"));
    }

    public Real setDigitsPrecision(int digitsPrecision) {
        context = BigFloat.context(digitsPrecision);
        value = context.valueOf(value);
        return this;
    }

    public int getDigitsPrecision() {
        return context.getPrecision();
    }

    public boolean isZero() {
        return BigFloat.abs(value).isLessThan(EPSILON);
    }

    public Real ln() {
        BigFloat x = context.valueOf(BigDecimalMath.log(value.toBigDecimal(), context.getMathContext()));
        return new Real(x, context.getMathContext());
    }

    public Real pow(double power) {
        BigFloat x = context.valueOf(BigDecimalMath.pow(value.toBigDecimal(), BigDecimal.valueOf(power), context.getMathContext()));
        return new Real(x, context.getMathContext());
    }

    public Real sqrt() {
        return pow(0.5);
    }

    public Real subtract(@NotNull Real v1) {
        return new Real(value.subtract(v1.value).toBigDecimal(), context);
    }

    public Real divide(@NotNull Real v1) {
        BigFloat dividend = value.divide(context.valueOf(v1.value));
        return new Real(dividend, context.getMathContext());
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

    public boolean isNotEqual(Object obj) {
        return !equals(obj);
    }

    @Override
    public int hashCode() {
        return value.hashCode();
    }

    public boolean isNegative() {
        return isLessThan(ZERO);
    }

    public boolean isLessThan(@NotNull Real real) {
        return value.compareTo(real.value) < 0;
    }

    public boolean isGreaterThan(@NotNull Real real) {
        return value.compareTo(real.value) > 0;
    }

    public boolean isPositive() {
        return value.compareTo(ZERO.value) > 0;
    }

    public static Real valueOf(double v) {
        return new Real(v, new MathContext(100));
    }

    public double toDouble() {
        return value.toDouble();
    }

    @Override
    public String toString() {
        return value.toString();
    }

    @Override
    public Real times(@NotNull Real real) {
        return new Real(value.multiply(real.value), context.getMathContext());
    }

    @Override
    public Real inverse() {
        return ONE.divide(this);
    }

    @Override
    public Real plus(@NotNull Real real) {
        return new Real(value.add(real.value), context.getMathContext());
    }

    @Override
    public Real opposite() {
        return ZERO.subtract(this);
    }

    @Override
    public Object copy() {
        return new Real(value, context.getMathContext());
    }
}
