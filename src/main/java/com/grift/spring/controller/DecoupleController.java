package com.grift.spring.controller;

import com.grift.model.Tick;
import com.grift.spring.service.DecoupleService;
import org.jetbrains.annotations.NotNull;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.context.annotation.Scope;
import org.springframework.context.annotation.ScopedProxyMode;
import org.springframework.http.MediaType;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestMethod;
import org.springframework.web.bind.annotation.RestController;

@RequestMapping("/api/tick")
@RestController
@Scope(value = "request", proxyMode = ScopedProxyMode.TARGET_CLASS)
public class DecoupleController {

    @Autowired
    DecoupleService decoupleService;

    @NotNull
    @RequestMapping(method = RequestMethod.POST, produces = {MediaType.APPLICATION_JSON_VALUE})
    public Tick insertOrUpdateTick(@NotNull @RequestBody Tick tick) {
        decoupleService.insertTick(tick);
        return tick;
    }
}
