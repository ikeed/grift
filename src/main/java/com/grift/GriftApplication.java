package com.grift;

import java.util.List;
import com.google.common.collect.Lists;
import com.grift.forex.symbol.ImmutableSymbolIndexMap;
import com.grift.forex.symbol.SymbolIndexMap;
import com.grift.math.decoupler.DecouplerMatrixColtImpl;
import com.grift.math.decoupler.Factory;
import com.grift.spring.service.DecoupleService;
import org.jetbrains.annotations.NotNull;
import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.context.annotation.Bean;

@SpringBootApplication
public class GriftApplication {
    private final List<String> symbolList = Lists.newArrayList("CAD", "USD", "GBP");

    @NotNull
    @Bean
    ImmutableSymbolIndexMap symbolIndexMap() {
        return new SymbolIndexMap().addSymbols(symbolList).getImmutablecopy();
    }

    @NotNull
    @Bean
    Factory decouplerFactory() {
        return new DecouplerMatrixColtImpl.Factory(symbolIndexMap());
    }

    @NotNull
    @Bean
    DecoupleService decouplerService() {
        return new DecoupleService(decouplerFactory());
    }

    public static void main(String[] args) {
        SpringApplication.run(GriftApplication.class, args);
    }
}
