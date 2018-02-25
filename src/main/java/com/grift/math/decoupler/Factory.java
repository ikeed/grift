package com.grift.math.decoupler;

import org.jetbrains.annotations.NotNull;

public interface Factory {
    @NotNull
    DecouplerMatrix make();
}
